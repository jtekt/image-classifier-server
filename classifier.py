import tensorflow as tf
from tensorflow import keras
import numpy as np
from fastapi import HTTPException
from os import getenv, path, listdir
import onnx
import onnxruntime
from dotenv import load_dotenv
from time import time
import json
import io
from glob import glob
import mlflow
from PIL import Image
from config import mlflow_tracking_uri, provider, warm_up
import traceback

load_dotenv()

if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

class Classifier:
    
    def __init__(self):
        self.model_path = "./model"
        self.models = {}
        self.model_loaded = False
        # モデルに関する追加情報を保持する属性
        self.model_infos = {}

        self.mlflow_model = None

        # 環境変数を使ってモデルパラメータを設定
        if getenv('CLASS_NAMES'):
            print('環境変数からクラス名を設定')
            self.class_names = getenv("CLASS_NAMES").split(',')

        # model_path内の各サブディレクトリからモデルをロード
        self.load_models()

    def load_models(self):
        subdirs = [d for d in listdir(self.model_path) if path.isdir(path.join(self.model_path, d))]
        for subdir in subdirs:
            model_dir = path.join(self.model_path, subdir)
            if glob(path.join(model_dir, "*")):
                try:
                    model, model_info = self.load_model_from_local(model_dir)
                    self.models[subdir] = model
                    self.model_infos[subdir] = model_info
                except Exception as e:
                    print(f'[AI] {model_dir} からモデルのロードに失敗')
                    print(e)
                    print(traceback.format_exc())
            elif mlflow_tracking_uri and getenv('MLFLOW_MODEL_VERSION') and getenv('MLFLOW_MODEL_NAME'):
                try:
                    self.load_model_from_mlflow(getenv('MLFLOW_MODEL_NAME'), getenv('MLFLOW_MODEL_VERSION'))
                except Exception as e:
                    print('[AI] MLflowからのモデルのロードに失敗')
                    print(e)

    def read_model_info(self, model_dir):
        file_path = path.join(model_dir, 'modelInfo.json')
        with open(file_path, 'r') as openfile:
            return json.load(openfile)

    def load_model_from_mlflow(self, model_name, model_version):
        # mlflowから任意の形式のモデルをロード
        self.model_info = {}
        
        print(f'[AI] MLflowの {mlflow_tracking_uri} からモデル {model_name} v{model_version} をダウンロード')

        self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        self.model_info['mlflow_url'] = f'{mlflow_tracking_uri}/#/models/{model_name}/versions/{model_version}'
        self.model_loaded = True
        self.model_info['origin'] = "mlflow"
        
        print('[AI] モデルがロードされました')
        
        if warm_up:
            self.warm_up()
    
    def load_model_from_local(self, model_dir):
        # ローカルディレクトリからモデルをロード
        model = None
        model_info = {}
        try:
            if glob(path.join(model_dir, "*.onnx")):
                self.model_name = path.basename(glob(path.join(model_dir, "*.onnx"))[0])
                model, model_info = self.load_model_from_onnx(model_dir)
            else:
                model, model_info = self.load_model_from_keras(model_dir)
                
            if warm_up:
                self.warm_up()
        except Exception as e:
            print('[AI] ローカルディレクトリからのモデルロードに失敗')
            print(e)
            print(traceback.format_exc())
        return model, model_info
    
    def load_model_from_keras(self, model_dir):
        print('[AI_keras] ロード中:', model_dir)
        
        model_info = {}
        model = keras.models.load_model(model_dir)
        
        self.model_loaded = True
        model_info['origin'] = "folder"
        model_info['type'] = "keras"

        # .jsonファイルからモデル情報を取得
        try:
            json_model_info = self.read_model_info(model_dir)
            model_info = {**model_info, **json_model_info}
        except:
            print('JSONファイルからのモデル情報のロードに失敗')

        print('[AI] モデルがロードされました')
        return model, model_info
        
    def load_model_from_onnx(self, model_dir):
        model_info = {}
        print('[AI_onnx] ロード中:', model_dir)

        file_path = path.join(model_dir, self.model_name)
        if not path.isfile(file_path):
            raise ValueError(f"モデルファイル {file_path} が存在しません")
        
        # onnxruntimeのプロバイダーを設定
        available_providers = onnxruntime.get_available_providers()
        
        if provider in available_providers:
            providers = [provider]
        else:
            providers = available_providers
        
        # onnxをGPUで推論するときの設定
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # セッションオプションを設定して、スレッド数を明示的に指定
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1  # 必要に応じてスレッド数を調整

        model = onnxruntime.InferenceSession(file_path, sess_options=session_options, providers=providers)
        
        model_info['origin'] = "folder"
        model_info['type'] = "onnx"
        model_info['providers'] = providers

        print('[AI] モデルがロードされました')
        print(f'[AI] ONNX ランタイムプロバイダー: {str(providers)}')

        return model, model_info
        
    def get_target_size(self, model):
        # 入力サイズの取得方法に応じて分ける
        if hasattr(model, 'input'):
            target_size = (model.input.shape[1], model.input.shape[2])

        elif hasattr(model, 'metadata'):
            input_shape = model.metadata.signature.inputs.to_dict()[0]['tensor-spec']['shape']
            target_size = (input_shape[1], input_shape[2])
            
        elif hasattr(model, 'get_inputs'):
            input_shape = model.get_inputs()[0].shape
            target_size = (input_shape[1], input_shape[2])
        return target_size


    async def load_image_from_request(self, file,model_name):
        import cv2
        fileBuffer = io.BytesIO(file)

        # モデルのターゲットサイズを取得
        self.target_size = self.get_target_size(self.models[model_name])
        
        resized_images = np.array([cv2.resize(img, self.target_size) for img in file])
        
        
        print("resized_image_shape:",resized_images.shape)
        # img = keras.preprocessing.image.load_img(fileBuffer, target_size=self.target_size)
        # img_array = keras.preprocessing.image.img_to_array(img)

        # バッチ軸を作成
        return resized_images
        # return tf.expand_dims(img_array, 0).numpy()

        def get_class_name(self, prediction, model_info):
            # 出力に名前を付ける
            max_index = np.argmax(prediction)
            return model_info['class_names'][max_index]
        
        def warm_up(self):
            # make dummy data
            self.get_target_size()
            input_ = np.ones(self.target_size, dtype='float32')
            num_pil = Image.fromarray(input_)
            num_byteio = io.BytesIO()
            num_pil.save(num_byteio, format='png')
            num_bytes = num_byteio.getvalue()
            

            initial_startup_time_start = time()
            # reshape dummy data
            fileBuffer = io.BytesIO(num_bytes)
            img = keras.preprocessing.image.load_img(fileBuffer, target_size=self.target_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            # Create batch axis
            model_input = tf.expand_dims(img_array, 0).numpy()
            # predict
            if hasattr(self.model, 'predict'):
                model_output = self.model.predict(model_input)
            elif hasattr(self.model, 'run'):
                output_names = [outp.name for outp in self.model.get_outputs()]
                input = self.model.get_inputs()[0]
                model_output = self.model.run(output_names, {input.name: model_input})[0]
            # Separate by type of output
            if isinstance(model_output, dict):
                prediction = model_output['pred'][0]
            else:
                prediction = model_output[0]
            initial_startup_time = time() - initial_startup_time_start
            print('[AI] The initial startup of model is done.')
            print('[AI] Initial startup time:', initial_startup_time, 's')
        
            
    async def predict_batch(self, image_list, model_name, process_info):
        if model_name not in self.models:
            raise ValueError(f"モデル {model_name} が見つかりません")

        model = self.models[model_name]
        model_info = self.model_infos[model_name]

        image_list = await self.load_image_from_request(image_list, model_name)

        # 画像の形状を確認
        image_list = np.array(image_list)
        if image_list.shape[1:] != (224, 224, 3):
            raise ValueError(f"入力画像の形状が正しくありません: {image_list.shape}")

        image_list = image_list.astype('float32')
        image_list /= 255.0

        # バッチ全体を一度に推論
        model_output = await self.predict(image_list, model_name)

        print("モデルの出力:", model_output)

        # prediction と process_info を集約
        # 'prediction'と'process_info'は行列形式で配置
        max_row = max([info['row'] for info in process_info])
        max_col = max([info['col'] for info in process_info])
        
        response = {
            'prediction': [[] for _ in range(max_row)],  # 行数に合わせて空のリストを準備
            'process_info': [[] for _ in range(max_row)],
            'inference_time': model_output['inference_time']  # バッチ全体に対して1つの推論時間
        }

        # 各画像の予測と process_info を row, col に従って収集
        for i, info in enumerate(process_info):
            row = info['row'] - 1  # 1-based indexを0-basedに変更
            col = info['col'] - 1  # 1-based indexを0-basedに変更
            
            # prediction と process_info の位置を揃える
            response['prediction'][row].append(model_output['prediction'][i])
            response['process_info'][row].append(info)

        return response

  
    
    # async def predict_batch(self, image_list, model_name):
    #     if model_name not in self.models:
    #         raise ValueError(f"モデル {model_name} が見つかりません")

    #     model = self.models[model_name]
    #     model_info = self.model_infos[model_name]
        
        
    #     image_list = await self.load_image_from_request(image_list,model_name)

    #     # 画像リストの形状が正しいことを確認
    #     image_list = np.array(image_list)
    #     if image_list.shape[1:] != (224, 224, 3):
    #         raise ValueError(f"入力画像の形状が正しくありません: {image_list.shape}")
        
        
    #     image_list = image_list.astype('float32')
    #     image_list /= 255.0

    #     # バッチ全体を一度に推論
    #     model_output = await self.predict(image_list, model_name)

    #     print("モデルの出力:", model_output)

    #     predictions = []

    #     # 各画像の予測結果を追加
    #     for i in range(len(image_list)):
    #         prediction_result = {
    #             'prediction': model_output['prediction'][i],


    async def predict(self, file, model_name):
        if model_name not in self.models:
            raise ValueError(f"モデル {model_name} が見つかりません")

        model = self.models[model_name]
        model_info = self.model_infos[model_name]

        # model_input = await self.load_image_from_request(file,model_name)

        print("モデル:", model)
        print("モデル情報:", model_info)

        inference_start_time = time()
        model_input = file

        model_output = None

        # 既存の関数に応じて分ける
        if hasattr(model, "predict"):
            model_output = model.predict(model_input)
        elif hasattr(model, "run"):
            output_names = [outp.name for outp in model.get_outputs()]
            input = model.get_inputs()[0]
            model_output = model.run(output_names, {input.name: model_input})[0]
            print(output_names)

        # model_outputがNoneでないことを確認する
        if model_output is None:
            raise ValueError(
                "モデルの出力がNoneです。モデルの推論が正しく実行されませんでした。"
            )

        inference_time = time() - inference_start_time

        print(model_output)
        response = {
            "prediction": model_output.tolist(),
            "inference_time": inference_time,
        }

        # クラス名が利用可能な場合は追加
        if "class_names" in model_info:
            response["class_names"] = [
                self.get_class_name(pred, model_info) for pred in model_output
            ]

        return response

import pathlib
import time

import numpy as np
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
import onnxruntime
import faiss

from config import (
    provider,
    faiss_no_use_gpu,
    PROV_TRT,
    PROV_VINO,
)


FLAVOR_NAME = "mlflow_patchcore"


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.
    """
    return _OnnxModelWrapper(path)


class _OnnxModelWrapper:
    def __init__(self, path):
        # Get the model meta data from the MLModel yaml file.
        local_path = pathlib.Path(path).parent
        model_meta = Model.load(str(local_path / MLMODEL_FILE_NAME))
        self.metadata = model_meta

        start_tm = time.time()
        opt = onnxruntime.SessionOptions()
        opt.intra_op_num_threads = 2

        providers = onnxruntime.get_available_providers()
        if provider in providers:
            providers = [provider]

        if PROV_TRT in providers:
            providers[providers.index(PROV_TRT)] = (PROV_TRT, {"trt_fp16_enable": True})
        if PROV_VINO in providers:
            if tuple([int(v) for v in onnxruntime.__version__.split(".")]) > (1, 17, 3):
                providers[providers.index(PROV_VINO)] = (PROV_VINO, {"device_type": "GPU"})
            else:
                providers[providers.index(PROV_VINO)] = (PROV_VINO, {"device_type": "GPU_FP32"})

        self.rt = onnxruntime.InferenceSession(path, sess_options=opt, providers=providers)
        print(f'[onnx] model load finish. provider={providers}', flush=True)

        self.inputs = [(inp.name, inp.type) for inp in self.rt.get_inputs()]
        self.output_names = [outp.name for outp in self.rt.get_outputs()]

        feat_path = str(local_path / model_meta.flavors.get(FLAVOR_NAME)['feat_name'])
        self.search_index = faiss.read_index(feat_path)
        print('[Faiss] read index finish', flush=True)
        if faiss_no_use_gpu == False:
            try:
                self.search_index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.search_index, faiss.GpuClonerOptions()
                )
                print('[Faiss] use GPU', flush=True)
            except:
                print('[Faiss] cannot find GPU', flush=True)

        # parameter
        self.n_nearest_neighbours = 1
        self.patchcore_threshold = float(model_meta.metadata.get('patchcore_threshold', '1.0'))
        self.patchcore_index = model_meta.metadata.get('patchcore_class_index', 0)

        """
        # dummy predict
        input_shape = model_meta.flavors.get(FLAVOR_NAME)['input_shape']
        dummy_data = np.zeros((1, *input_shape, 3), dtype=np.float32)
        _ = self.rt.run(self.output_names, {self.inputs[0][0]: dummy_data})
        """

        print(f"[PATCHCORE] load time: {time.time() - start_tm}")

    def predict(self, data):
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, dict):
            data = list(data.values())[0]
        else:
            raise TypeError(
                "Input should be a dictionary or a numpy array, "
                f"got '{type(data)}'"
            )

        bsize = 32
        predicts = []
        scores   = []
        masks    = []
        for idx in range(0, len(data), bsize):
            feed_dict = {self.inputs[0][0]: data[idx:idx+bsize]}
            feature, pred = self.rt.run(self.output_names, feed_dict)
            predicts.append(pred)

            f_shape = feature.shape
            feature = np.reshape(feature, (-1, f_shape[-1]))
            dists, _ = self.search_index.search(feature, self.n_nearest_neighbours)
            dists = np.mean(dists, axis=-1)

            mask = np.reshape(dists, (*f_shape[:-1],))
            masks.append(mask)

            score = np.reshape(dists, (f_shape[0], -1))
            score = np.amax(score, axis=1)
            scores.append(score)

        predicts = np.concatenate(predicts)
        scores   = np.concatenate(scores)
        masks    = np.concatenate(masks)

        scores_norm = np.clip(0.5 * scores / self.patchcore_threshold, 0.0, 1.0)

        return predicts, scores, scores_norm, masks


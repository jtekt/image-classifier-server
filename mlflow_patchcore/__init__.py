# import os
# import sys
# import yaml
# from pathlib import Path
# from packaging.version import Version
# from typing import Any, Dict, Optional
# import time

# import numpy as np
# from mlflow import pyfunc
# from mlflow.models import Model, ModelInputExample, ModelSignature, infer_pip_requirements
# from mlflow.models.model import MLMODEL_FILE_NAME
# from mlflow.exceptions import MlflowException
# from mlflow.utils.environment import (
#     _mlflow_conda_env,
#     _process_pip_requirements,
#     _CONDA_ENV_FILE_NAME,
#     _REQUIREMENTS_FILE_NAME,
#     _PYTHON_ENV_FILE_NAME,
#     _PythonEnv,
# )
# from mlflow.utils.requirements_utils import _get_pinned_requirement
# from mlflow.utils.file_utils import write_to
# from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
# from mlflow.utils.model_utils import _validate_and_prepare_target_save_path
# from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
# import onnxruntime

# import faiss
# from config import provider, warm_up, faiss_no_use_gpu


# FLAVOR_NAME = "mlflow_patchcore"
# onnx_model_filename = "model.onnx"
# patchcore_data_filename = "fais_nn"


# def get_default_pip_requirements():
#     """
#     :return: A list of default pip requirements for MLflow Models produced by this flavor.
#              Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
#              that, at minimum, contains these requirements.
#     """
#     return list(
#         map(
#             _get_pinned_requirement,
#             [
#                 "onnx",
#                 # The ONNX pyfunc representation requires the OnnxRuntime
#                 # inference engine. Therefore, the conda environment must
#                 # include OnnxRuntime
#                 #"onnxruntime-gpu",
#                 "faiss",
#             ],
#         )
#     )


# def get_default_conda_env():
#     """
#     :return: The default Conda environment for MLflow Models produced by calls to
#              :func:`save_model()` and :func:`log_model()`.
#     """
#     return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


# @format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
# def log_model(
#     onnx_model,
#     search_index,
#     artifact_path,
#     signature: ModelSignature = None,
#     metadata = None,
# ):
#     """
#     :param onnx_model: ONNX model to be saved.
#     :param search_index: faiss search_index
#     :param artifact_path: Run-relative artifact path.
#     :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
#              metadata of the logged model.
#     """
#     new_metadata = {
#         'evaluate_output_pt': 1,
#         'output_list': ['DNN', 'PATCHCORE'],
#     }
#     if metadata is not None:
#         if isinstance(metadata, dict):
#             new_metadata.update(metadata)
#         else:
#             raise Exception(f'ERROR: "metadata" must be dictionary.')

#     return Model.log(
#         artifact_path=artifact_path,
#         flavor=sys.modules[__name__],
#         onnx_model=onnx_model,
#         search_index=search_index,
#         signature=signature,
#         await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
#         metadata=new_metadata,
#     )


# @format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
# def save_model(
#     onnx_model,
#     search_index,
#     path,
#     mlflow_model=None,
#     signature=None,
# ):
#     """
#     :param onnx_model: ONNX model to be saved.
#     :param path: Local path where the model is to be saved.
#     :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
#     """
#     import onnx
#     import glob

#     path = os.path.abspath(path)
#     _validate_and_prepare_target_save_path(path)

#     if mlflow_model is None:
#         mlflow_model = Model()
#     if signature is not None:
#         mlflow_model.signature = signature
#     model_data_subpath = onnx_model_filename
#     model_data_path = os.path.join(path, model_data_subpath)

#     # Save onnx-model
#     if Version(onnx.__version__) >= Version("1.9.0"):
#         onnx.save_model(onnx_model, model_data_path, save_as_external_data=True)
#     else:
#         onnx.save_model(onnx_model, model_data_path)

#     feat_name = patchcore_data_filename
#     feat_filename = os.path.join(path, feat_name)
#     faiss.write_index(search_index, feat_filename)

#     input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim[1:3]]

#     pyfunc.add_to_model(
#         mlflow_model,
#         loader_module=__name__,
#         data=model_data_subpath,
#         conda_env=_CONDA_ENV_FILE_NAME,
#         python_env=_PYTHON_ENV_FILE_NAME,
#     )

#     mlflow_model.add_flavor(
#         FLAVOR_NAME,
#         onnx_version=onnx.__version__,
#         data=model_data_subpath,
#         feat_name=feat_name,
#         input_shape=input_shape,
#     )
#     mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

#     default_reqs = get_default_pip_requirements()
#     # To ensure `_load_pyfunc` can successfully load the model during the dependency
#     # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
#     inferred_reqs = infer_pip_requirements(
#         path,
#         FLAVOR_NAME,
#         fallback=default_reqs,
#     )
#     default_reqs = sorted(set(inferred_reqs).union(default_reqs))
#     conda_env, pip_requirements, _ = _process_pip_requirements(default_reqs)

#     with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
#         yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

#     # Save `requirements.txt`
#     write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

#     _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


# def _load_pyfunc(path):
#     """
#     Load PyFunc implementation. Called by ``pyfunc.load_model``.
#     """
#     return _OnnxModelWrapper(path)


# class _OnnxModelWrapper:
#     def __init__(self, path):
#         # Get the model meta data from the MLModel yaml file.
#         local_path = str(Path(path).parent)
#         model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))
#         self.metadata = model_meta

#         start_tm = time.time()
#         #providers = ["TensorrtExecutionProvider"]
#         #providers = ["CUDAExecutionProvider"]
#         #providers = ["CPUExecutionProvider"]
#         providers = onnxruntime.get_available_providers()
#         if provider in providers:
#             providers = [provider]

#         self.rt = onnxruntime.InferenceSession(path, providers=providers)
#         print(f'[onnx] model load finish. provider={providers}', flush=True)

#         self.inputs = [(inp.name, inp.type) for inp in self.rt.get_inputs()]
#         self.output_names = [outp.name for outp in self.rt.get_outputs()]

#         feat_path = os.path.join(local_path, model_meta.flavors.get(FLAVOR_NAME)['feat_name'])
#         self.search_index = faiss.read_index(feat_path)
#         print('[Faiss] read index finish', flush=True)
#         if faiss_no_use_gpu == False:
#             try:
#                 self.search_index = faiss.index_cpu_to_gpu(
#                     faiss.StandardGpuResources(), 0, self.search_index, faiss.GpuClonerOptions()
#                 )
#                 print('[Faiss] use GPU', flush=True)
#             except:
#                 print('[Faiss] cannot find GPU', flush=True)

#         # parameter
#         self.n_nearest_neighbours = 1
#         self.patchcore_threshold = float(model_meta.metadata.get('patchcore_threshold', '1.0'))
#         self.patchcore_index = model_meta.metadata.get('patchcore_class_index', 0)

#         # dummy predict
#         input_shape = model_meta.flavors.get(FLAVOR_NAME)['input_shape']
#         dummy_data = np.zeros((1, *input_shape, 3), dtype=np.float32)
#         _, _ = self.rt.run(self.output_names, {self.inputs[0][0]: dummy_data})

#         print(f"[PATCHCORE] load time: {time.time() - start_tm}")

#     def predict(self, data):
#         if isinstance(data, np.ndarray):
#             pass
#         elif isinstance(data, dict):
#             data = list(data.values())[0]
#         else:
#             raise TypeError(
#                 "Input should be a dictionary or a numpy array, "
#                 f"got '{type(data)}'"
#             )

#         bsize = 32
#         predicts = []
#         scores   = []
#         masks    = []
#         for idx in range(0, len(data), bsize):
#             feed_dict = {self.inputs[0][0]: data[idx:idx+bsize]}
#             feature, pred = self.rt.run(self.output_names, feed_dict)
#             predicts.append(pred)

#             f_shape = feature.shape
#             feature = np.reshape(feature, (-1, f_shape[-1]))
#             dists, _ = self.search_index.search(feature, self.n_nearest_neighbours)
#             dists = np.mean(dists, axis=-1)

#             mask = np.reshape(dists, (*f_shape[:-1],))
#             masks.append(mask)

#             score = np.reshape(dists, (f_shape[0], -1))
#             score = np.amax(score, axis=1)
#             scores.append(score)

#         predicts = np.concatenate(predicts)
#         scores   = np.concatenate(scores)
#         masks    = np.concatenate(masks)

#         scores_norm = 0.5 * scores / self.patchcore_threshold
#         scores_norm[scores_norm > 1.0] = 1.0
#         #scores_norm = np.stack([scores_norm if i != self.patchcore_index else 1-scores_norm for i in range(predicts.shape[1])], axis=1)

#         return predicts, scores, scores_norm, masks

import os
import sys
import yaml
from pathlib import Path
from packaging.version import Version
from typing import Any, Dict, Optional
import time

import numpy as np
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.exceptions import MlflowException
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _process_pip_requirements,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.file_utils import write_to
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.model_utils import _validate_and_prepare_target_save_path
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
import onnxruntime

import faiss
from config import provider, warm_up, faiss_no_use_gpu


FLAVOR_NAME = "mlflow_patchcore"
onnx_model_filename = "model.onnx"
patchcore_data_filename = "fais_nn"


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return list(
        map(
            _get_pinned_requirement,
            [
                "onnx",
                # The ONNX pyfunc representation requires the OnnxRuntime
                # inference engine. Therefore, the conda environment must
                # include OnnxRuntime
                #"onnxruntime-gpu",
                "faiss",
            ],
        )
    )


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    onnx_model,
    search_index,
    artifact_path,
    signature: ModelSignature = None,
    metadata = None,
):
    """
    :param onnx_model: ONNX model to be saved.
    :param search_index: faiss search_index
    :param artifact_path: Run-relative artifact path.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    new_metadata = {
        'evaluate_output_pt': 1,
        'output_list': ['DNN', 'PATCHCORE'],
    }
    if metadata is not None:
        if isinstance(metadata, dict):
            new_metadata.update(metadata)
        else:
            raise Exception(f'ERROR: "metadata" must be dictionary.')

    return Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        onnx_model=onnx_model,
        search_index=search_index,
        signature=signature,
        await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        metadata=new_metadata,
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    onnx_model,
    search_index,
    path,
    mlflow_model=None,
    signature=None,
):
    """
    :param onnx_model: ONNX model to be saved.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    """
    import onnx
    import glob

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    model_data_subpath = onnx_model_filename
    model_data_path = os.path.join(path, model_data_subpath)

    # Save onnx-model
    if Version(onnx.__version__) >= Version("1.9.0"):
        onnx.save_model(onnx_model, model_data_path, save_as_external_data=True)
    else:
        onnx.save_model(onnx_model, model_data_path)

    feat_name = patchcore_data_filename
    feat_filename = os.path.join(path, feat_name)
    faiss.write_index(search_index, feat_filename)

    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim[1:3]]

    pyfunc.add_to_model(
        mlflow_model,
        loader_module=__name__,
        data=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        onnx_version=onnx.__version__,
        data=model_data_subpath,
        feat_name=feat_name,
        input_shape=input_shape,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    default_reqs = get_default_pip_requirements()
    # To ensure `_load_pyfunc` can successfully load the model during the dependency
    # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
    inferred_reqs = infer_pip_requirements(
        path,
        FLAVOR_NAME,
        fallback=default_reqs,
    )
    default_reqs = sorted(set(inferred_reqs).union(default_reqs))
    conda_env, pip_requirements, _ = _process_pip_requirements(default_reqs)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.
    """
    return _OnnxModelWrapper(path)


class _OnnxModelWrapper:
    def __init__(self, path):
        # Get the model meta data from the MLModel yaml file.
        local_path = str(Path(path).parent)
        model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))
        self.metadata = model_meta

        start_tm = time.time()
        opt = onnxruntime.SessionOptions()
        opt.intra_op_num_threads = 2

        #providers = ["TensorrtExecutionProvider"]
        #providers = ["CUDAExecutionProvider"]
        #providers = ["CPUExecutionProvider"]
        providers = onnxruntime.get_available_providers()
        if provider in providers:
            providers = [provider]

        self.rt = onnxruntime.InferenceSession(path, sess_options=opt, providers=providers)
        print(f'[onnx] model load finish. provider={providers}', flush=True)

        self.inputs = [(inp.name, inp.type) for inp in self.rt.get_inputs()]
        self.output_names = [outp.name for outp in self.rt.get_outputs()]

        feat_path = os.path.join(local_path, model_meta.flavors.get(FLAVOR_NAME)['feat_name'])
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

        # dummy predict
        input_shape = model_meta.flavors.get(FLAVOR_NAME)['input_shape']
        dummy_data = np.zeros((1, *input_shape, 3), dtype=np.float32)
        _, _ = self.rt.run(self.output_names, {self.inputs[0][0]: dummy_data})

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

        scores_norm = 0.5 * scores / self.patchcore_threshold
        scores_norm[scores_norm > 1.0] = 1.0
        #scores_norm = np.stack([scores_norm if i != self.patchcore_index else 1-scores_norm for i in range(predicts.shape[1])], axis=1)

        return predicts, scores, scores_norm, masks

"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
from ood_enabler.model_wrapper.tf import TFWrapper
from ood_enabler.model_wrapper.pytorch import PytorchWrapper
from ood_enabler.model_wrapper.onnx import ONNXWrapper
from ood_enabler.util.constants import MLBackendType
from ood_enabler.exceptions.exceptions import UnknownMLBackend


def get_wrapper(model_metadata, model_file, **kwargs):
    """
    returns a ModelWrapper based on the underlying model to enable with OOD

    :param model_metadata: metadata about the model, included type, and architecture
    :type model_metadata: `dict`
    :param model_file: path to local model_file
    :type model_file: `str`
    :return: An instantiated ModelWrapper
    """
    # TODO: architecture needs to be taken into account; currently only supports resnet50 for all backends

    if model_metadata['type'] == MLBackendType.PYTORCH:
        return PytorchWrapper(model_metadata, path=model_file, **kwargs)

    elif model_metadata['type'] == MLBackendType.TF:
        return TFWrapper(model_metadata, path=model_file, **kwargs)

    elif model_metadata['type'] == MLBackendType.ONNX:
        return ONNXWrapper(model_metadata, path=model_file, **kwargs)


    else:
        raise UnknownMLBackend("Model type not recognized or supported")
from ood_enabler.model_modifier.pytorch_modifier import PytorchModifier
from ood_enabler.model_modifier.tf_modifier import TFModifier
from ood_enabler.model_modifier.onnx_modifier import ONNXModifier
from ood_enabler.util.constants import MLBackendType, SupportedModelArch
from ood_enabler.exceptions.exceptions import UnknownMLBackend


def get_modifier(model_metadata):
    """
    returns a ModelTransformer based on the underlying model to enable with OOD

    :param model_metadata: metadata about the model, included type, and architecture
    :type model_metadata: `dict`
    :return: An instantiated ModelTransformer
    """
    # TODO: architecture needs to be taken into account; currently only supports resnet50 for all backends
    if model_metadata['type'] == MLBackendType.PYTORCH.value or model_metadata['type'] == MLBackendType.PYTORCH:
        return PytorchModifier()

    elif model_metadata['type'] == MLBackendType.TF.value or model_metadata['type'] == MLBackendType.TF:
        return TFModifier()

    elif model_metadata['type'] == MLBackendType.ONNX.value or model_metadata['type'] == MLBackendType.ONNX:
        return ONNXModifier()

    else:
        raise UnknownMLBackend

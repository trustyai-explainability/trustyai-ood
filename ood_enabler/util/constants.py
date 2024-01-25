"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
from enum import Enum


class MLBackendType(str, Enum):
    PYTORCH = 'pytorch'
    TF = 'tf'
    ONNX = 'onnx'


class SupportedModelArch(str, Enum):
    RESNET50 = 'resnet50'


class InferenceServiceType(str, Enum):
    IN_MEMORY = 'in-memory'
    KSERVE = 'kserve'
    RHODS = 'rhods'


class SavedModelFormat(str, Enum):
    NATIVE = 'native'
    TORCHSERVE = 'torchserve'
    TORCHSCRIPT = 'torchscript'
    STATE_DICT = 'state_dict'
    ONNX = 'onnx'


MAP_TO_NATIVE_DICT = {
    MLBackendType.PYTORCH: {
        SupportedModelArch.RESNET50: 'resnet50'
    },
    MLBackendType.TF: {
        SupportedModelArch.RESNET50: 'ResNet50'
    },
}

/*
 * © Copyright IBM Corp. 2024, and/or its affiliates. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the “License”);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

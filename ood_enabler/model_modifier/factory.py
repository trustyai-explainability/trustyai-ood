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

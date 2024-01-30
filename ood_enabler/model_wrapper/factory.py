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
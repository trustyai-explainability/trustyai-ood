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
from ood_enabler.inference_service.inmemory_inference import InMemoryInference
from ood_enabler.inference_service.kserve_inference import KserveInference
from ood_enabler.inference_service.rhods_inference import RhodsInference

from ood_enabler.util.constants import InferenceServiceType
from ood_enabler.exceptions.exceptions import UnknownInferenceService


def get_inference_service(service, **kwargs):
    """
    Returns an InferenceService based on the provided service, along with model to infer
    :param service: The type of inference service to use: in-memory | kserve
    :type service: `str`
    :return: An instantiated InferenceService
    """
    if service == InferenceServiceType.KSERVE:
        return KserveInference(**kwargs)

    elif service == InferenceServiceType.RHODS:
        return RhodsInference(**kwargs)

    elif service == InferenceServiceType.IN_MEMORY:
        return InMemoryInference(**kwargs)

    else:
        # raise exception or return IN_MEMORY?
        # raise UnknownInferenceService("Unknown inference service provided")
        return InMemoryInference(**kwargs)

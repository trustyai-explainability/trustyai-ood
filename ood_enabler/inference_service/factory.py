"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
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

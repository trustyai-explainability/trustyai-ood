from ood_enabler.util.constants import InferenceServiceType
from ood_enabler.model_modifier.factory import get_modifier
from ood_enabler.inference_service.factory import get_inference_service


class OODEnabler:
    """
    Class for enabling a model with OOD capabilities
    """

    # model_modifier doesn't need to be passed as argument; can be built from model wrapper metadata
    @staticmethod
    def ood_enable(model_wrapper, data_handler=None, inference_service=InferenceServiceType.IN_MEMORY, **kwargs):
        """
        Enables a model with OOD capabilities (i.e. inserting ood/normalization layer in model)

       :param model_wrapper: model to be enabled
       :type model_wrapper: `model_wrapper.Model`
       :param data_handler: in-distribution data for normalizing ood layer (optional)
       :type data_handler: `data.DataHandler`
       :param inference_service: mode for performing inference if normalizing data
       :return:
       """

        model_modifier = get_modifier(model_wrapper.model_metadata)
        model_modifier.add_ood_layer(model_wrapper)

        if data_handler:
            inference_service = get_inference_service(inference_service, **kwargs)
            inference_results = inference_service.infer(model_wrapper, data_handler)
            model_modifier.add_normalization_layer(model_wrapper, inference_results)

        return model_wrapper

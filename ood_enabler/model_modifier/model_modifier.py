from abc import ABC, abstractmethod


class ModelModifier(ABC):
    """
    Base class for Model Transformer classes, which transforms by embedding an OOD layer into model
    """

    @abstractmethod
    def add_ood_layer(self, model):
        """
        Based on input method, add OOD layer to the model

        :param model: model to embed with OOD layer
        :type model: `model_wrapper.Model`
        return: transformed OOD Model
        """
        raise NotImplementedError

    @abstractmethod
    def add_normalization_layer(self, model, inference_results):
        """
        Based on inference results (forward pass), add normalization layer for OOD

        :param model: model to embed with normalization layer
        :type model: `model_wrapper.Model`
        :param inference_results: results from transformed OOD model; should include ood_score

        """
        raise NotImplementedError

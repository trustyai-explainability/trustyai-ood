"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
from abc import ABC, abstractmethod


class InferenceService(ABC):
    """
        Base class to perform inference on inlier dataset for normalizing OOD scores
    """
    @abstractmethod
    def infer(self, model, data):
        """
        Performs inference on provided model with dataset

        :param model: model to use for inference
        :type model: `model.ModelWrapper`
        :param data: data to be inferred
        :type data: `data.Datahandler`
        :return: stats for normalizing data
        """
        raise NotImplementedError

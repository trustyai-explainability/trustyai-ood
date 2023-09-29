from abc import ABC, abstractmethod


class ModelWrapper(ABC):
    """
    Base Wrapper Class for importing model and storing metadata
    """
    def __init__(self, model_metadata, model=None, path=None, **kwargs):
        """
        Initializes wrapper for model

        :param model_metadata: metadata about model (i.e. architecture, type)
        :type model_metadata: `dict`
        :param model: loaded model (only one of model and path needs to be specified)
        :param path: path to be loaded (only one of model and path needs to be specified)
        """
        self.model_metadata = model_metadata
        if model is not None:
            self.model = model
        elif path is not None:
            self.model = self.load(path, **kwargs)
        else:
            raise ValueError('Either model or path needs to be specified')

    @abstractmethod
    def load(self, path, bypass_model_check=False, **kwargs):
        """
        loads in memory a saved model of the type of the underlying wrapper

        :param path: path to where model is accessible
        :param bypass_model_check: if True, checking of whether the architecture is supported is disabled; can be useful for loading an OOD-enabled model for testing purpose
        :return: the loaded model
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path, saved_model_format):
        """
        saves underlying model to provided path

        :param path: path to where model should be saved
        :param saved_model_format: file/directory format of saved model
        :type saved_model_format: `util.constants.SavedModelFormat`
        :return: path to where model is saved
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, data_handler, func=None):
        """
        infers on underlying model

        :param data_handler: datahandler with to be inferred
        :type data_handler: `data.data_handler`
        :param func: function callback for inference; if None, use native inference call from ML backend
        :return: inference results
        """
        raise NotImplementedError

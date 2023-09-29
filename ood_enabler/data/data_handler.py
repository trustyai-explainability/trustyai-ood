from abc import ABC, abstractmethod


class DataHandler(ABC):
    """
    Base class to load and pre-process data
    """
    def __init__(self,  dataset=None):
        self._dataset = dataset
    
    @abstractmethod
    def _load_dataset(self, storage, source, destination, metadata):
        raise NotImplementedError
    
    def load_dataset(self, storage, source, destination, metadata):
        """
        downloads data from storage backend and loads for pre-processing
        and store a reference in this instance available via get_dataset().
        :param storage: a storage backend to download data from
        :type storage: `storage.Storage`
        :param source: location of image dataset in storage backend
        :type source: `str`
        :param destination: location to download_dataset locally
        :type destination: `str`
        :param metadata : metadata about dataset to be loaded
        :type metadata: `dict`
        """
        self._dataset = self._load_dataset(storage, source, destination, metadata)

    def get_dataset(self):
        """
        Get the most recent data loaded via load_dataset() or None
        if not called yet, or perhaps placed into this instance via the initializer
        """
        return self._dataset
"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
from abc import ABC, abstractmethod


class Storage(ABC):
    """
    Base class for accessing asset storage; can be cloud cloud or local filesystem
    """
    @abstractmethod
    def retrieve(self, source, destination):
        """
        Retrieves asset from provided source path and saves to destination

        :param source: path to asset
        :type source: `str`
        :param destination: path to store asset
        :return: path to saved file
        """
        raise NotImplementedError

    @abstractmethod
    def store(self, source, destination):
        """
        Stores asset from provided source path and saves to destination

        :param source: path
        :param destination:
        :return: path to uploaded file
        """
        raise NotImplementedError

    @abstractmethod
    def store_temporary(self, source, destination):
        """
        Stores asset from provided source path and saves to a temporary destination

        :param source: path
        :param destination:
        :return: an object that implements __exit__, to be called with 'with' statement
        """
        raise NotImplementedError

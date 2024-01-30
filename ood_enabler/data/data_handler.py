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
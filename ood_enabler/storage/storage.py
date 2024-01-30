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

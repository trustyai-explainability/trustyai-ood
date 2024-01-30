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
import os
import shutil
from ood_enabler.storage.storage import Storage
from ood_enabler.exceptions.exceptions import OODEnableException
from tempfile import TemporaryDirectory


class FileSystemStorage(Storage):
    """
    Class to store/retrieve assets from the local filesystem
    """
    def retrieve(self, source, destination):
        """
        Retrieves asset from provided source path and saves to destination

        :param source: path to asset
        :type source: `str`
        :param destination: path to store asset
        :return: path to saved file
        """
        if not os.path.exists(source):
            raise OODEnableException("File not found at specified source location")

        if not os.path.exists(destination):
            os.makedirs(destination)

        try:
            srcpath = os.path.abspath(source)
            srcdir = os.path.dirname(srcpath)
            if os.path.isdir(destination):
                destdir = os.path.abspath(destination)
                if destdir == srcdir:
                    return srcpath  # source and dest are identical.
            path = shutil.copy(source, destination)
            return path

        except IsADirectoryError:
            return shutil.copytree(src=source, dst=destination, dirs_exist_ok=True)

    def store(self, source, destination):
        """
        Stores asset from provided source path and saves to destination

        :param source: path to retrieve file
        :param destination: path to store file
        :return: path to exported file
        """
        return self.retrieve(source, destination)

    def store_temporary(self, source, destination=''):
        """
        Stores asset from provided source path and saves to a temporary directory

        :param source: path to retrieve file
        :param destination: path to store file (has no effect, only to match the function signature of base class).
        :return: a TemporaryDirectory object, to be called with 'with' statement
        """

        # Currently copying twice to temporary directory when using with ModelStore, once in ModelStore and once here.
        # The TemporaryDirectory object here is returned so that it can be used by the caller.
        tmp_dir = TemporaryDirectory()
        self.retrieve(source, tmp_dir.name)
        return tmp_dir

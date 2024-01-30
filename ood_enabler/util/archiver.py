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
import tarfile
import pathlib
import shutil
from zipfile import ZipFile
from ood_enabler.exceptions.exceptions import UnknownExtension


def archive(path, destination_path="."):
    """
    Archives a path to a tar.gz file in destination directory

    :param path: path to dir to archive
    :type path: `str`
    :param destination_path: path to save archived dir
    :type destination_path: `str`
    :return: path to archived dir
    """
    output_filename = os.path.join(destination_path, path.split('/')[-1])
    archive_file = shutil.make_archive(output_filename, 'gztar', path)
    return archive_file


def extract_archive(source_path, destination_path='.'):
    """
    Unarchives a zip or .tar.gz file

    :param source_path: path to .zip or .tarfile
    :type source_path: `str`
    :param destination_path: path to extract archive
    :type destination_path: `str`
    :return: path to extracted archive
    """
    if source_path.endswith('.zip'):
        with ZipFile(source_path, 'r') as zip_ref:
            extract_dir = os.path.join(destination_path, '{}'.format(pathlib.Path(source_path).with_suffix(''))
                                       .split('/')[-1])

            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)

        return extract_dir

    elif source_path.endswith('.tar.gz') or source_path.endswith('.tgz'):
        tar = tarfile.open(source_path, "r:gz")
        extract_dir = os.path.join(destination_path, '{}'.format(pathlib.Path(source_path).with_suffix(''))
                                   .split('/')[-1].split('.tar')[0])

        os.makedirs(extract_dir, exist_ok=True)
        tar.extractall(extract_dir)
        tar.close()

        return extract_dir

    else:
        raise UnknownExtension("Unsupported archive extension; should be .zip or .tar.gz")

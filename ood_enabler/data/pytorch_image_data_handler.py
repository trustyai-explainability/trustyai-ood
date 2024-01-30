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
import torch
from torchvision import transforms
import torchvision
from ood_enabler.data.data_handler import DataHandler
from ood_enabler.exceptions.exceptions import OODEnableException
from ood_enabler.util.archiver import extract_archive


class PytorchImageDataHandler(DataHandler):
    """
    Class for loading and preprocessing image dataset for Pytorch backend
    """
    def __init__(self, data=None):
        """
        Intializes datahander with in-memory dataset
        :param dataset: image dataset loaded into memory
        :type dataset: `torch.Tensor` or `torch.util.data.DataLoader`
        """
        super().__init__(data)

    def _load_dataset(self, storage, source, destination, metadata):
        """
        Downloads and loads a dataset from storage backend

        :param storage: storage backend connection
        :type storage: `storage.Storage`
        :param source: location of image dataset in storage backend
        :type source: `str`
        :param destination: location to download_dataset locally
        :type destination: `str`
        :param metadata: info about the image set (height, weight, batch_size and normalization (mean, and std)
        :type metadata: `dict`
        :return: loaded dataset
        :rtype tensorflow.python.data.ops.dataset_ops.BatchDataset
        """

        data_path = storage.retrieve(source, destination)

        if not os.path.isdir(data_path):
            # assume downloaded dataset is .tar.gz or zipped
            data_path = extract_archive(data_path, destination)

        transforms_list = [transforms.Resize((metadata['img_height'], metadata['img_width'])),
                           transforms.ToTensor()]
        if 'normalize' in metadata:
            transforms_list.append(transforms.Normalize(metadata['normalize'][0], metadata['normalize'][1]))

        img_transform = transforms.Compose(transforms_list)

        dataset = torchvision.datasets.ImageFolder(root=data_path, transform=img_transform)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=metadata.get('batch_size', 32))

        return dataloader

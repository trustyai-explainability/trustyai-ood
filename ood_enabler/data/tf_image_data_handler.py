import os
import numpy as np
import tensorflow as tf
from ood_enabler.data.data_handler import DataHandler
from ood_enabler.exceptions.exceptions import OODEnableException
from ood_enabler.util.archiver import extract_archive


class TFImageDataHandler(DataHandler):
    """
    Class for loading and preprocessing image dataset for TF backend
    """
    def __init__(self, data=None):
        """
        Intializes datahander with in-memory dataset
        :param data: image dataset loaded into memory
        :type dataset: `np.ndarray` or `tensorflow.python.data.ops.dataset_ops.BatchDataset`
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
        :param metadata: info about the image set (height, width, batch_size)
        :type metadata: `dict`
        :return: loaded dataset
        :rtype tensorflow.python.data.ops.dataset_ops.BatchDataset
        """

        data_path = storage.retrieve(source, destination)

        if not os.path.isdir(data_path):
            # assume downloaded dataset is .tar.gz or zipped
            data_path = extract_archive(data_path, destination)

        dataset = tf.keras.utils.image_dataset_from_directory(
            data_path,
            labels=None,
            image_size=(metadata['img_height'], metadata['img_width']),
            batch_size=metadata.get('batch_size', 32))

        if 'normalize' in metadata:
            dataset = self._normalize(dataset, metadata['normalize'])

        return dataset

    def _normalize(self, dataset, rescale_value):
        """
        Standardizes dataset by rescaling to specified value (i.e. 1./rescale_value

        :param dataset: the dataset to normalize
        :type dataset: `tensorflow.python.data.ops.dataset_ops.BatchDataset`
        :param rescale_value: value to rescale dataset
        :type rescale_value: `int`
        """
        normalization_layer = tf.keras.layers.Rescaling(1. / rescale_value)
        normalized_ds = dataset.map(lambda x: (normalization_layer(x)))
        return normalized_ds

    # def resize(self, img_height, img_width, **kwargs):
    #     resizing_layer = tf.keras.layers.Resizing(
    #         height=img_height,
    #         width=img_width,
    #         **kwargs)
    #
    #     resized_ds = self._dataset.map(lambda x: (resizing_layer(x)))
    #     self._dataset = resized_ds

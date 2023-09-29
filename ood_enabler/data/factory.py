from ood_enabler.data.pytorch_image_data_handler import PytorchImageDataHandler
from ood_enabler.data.tf_image_data_handler import TFImageDataHandler
from ood_enabler.util.constants import MLBackendType
from ood_enabler.exceptions.exceptions import UnknownMLBackend


def get_image_data_handler(model_backend, ds_metadata, local_store, source, destination='.'):
    """
    returns an image data_handler based on the underlying model
    :param model_backend: the underlying model backend type
    :type model_backend: `str`
    :param ds_metadata: Info about dataset for preprocessing specific to dataset type
    :type ds_metadata: `dict`
    :param local_store: storage pointing to where data set is located
    :type local_store: `storage.Storage`
    :source: path to find data in local store
    :type source: `str`
    :param destination: path to stored retrieved dataset before loading
    :type destination: `str`
    :return: An initialized data handler
    """
    if model_backend == MLBackendType.TF.value or model_backend == MLBackendType.TF:
        data_handler = TFImageDataHandler()
        data_handler.load_dataset(local_store, source, destination, ds_metadata)
        return data_handler

    elif model_backend == MLBackendType.PYTORCH.value or model_backend == MLBackendType.PYTORCH:
        data_handler = PytorchImageDataHandler()
        data_handler.load_dataset(local_store, source, destination, ds_metadata)
        return data_handler

    else:
        raise UnknownMLBackend

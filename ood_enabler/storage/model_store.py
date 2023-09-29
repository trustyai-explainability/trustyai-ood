import os
from tempfile import TemporaryDirectory
from ood_enabler.storage.local_storage import FileSystemStorage
from ood_enabler.storage.cos_storage import COSStorage
from ood_enabler.model_wrapper.factory import get_wrapper
from ood_enabler.util.constants import SavedModelFormat


class ModelStore:
    """
    Class responsible for uploading, downloading, and loading models from Storage backend
    """
    def __init__(self, storage):
        """
        Initializes a model store connection to a storage backend

        :param storage: An initialized storage backend
        :type storage: `storage.Storage`
        """
        self.storage = storage

    @staticmethod
    def from_filesystem():
        """
        Returns a ModelStore with FileSystem Storage

        :return: ModelStore
        """
        return ModelStore(FileSystemStorage())

    @staticmethod
    def from_cos(bucket, api_key=None, service_instance_id=None, endpoint=None, auth_endpoint=None):
        """
        Returns a ModelStore with FileSystem Storage

        :param bucket: COS store bucket
        :type bucket: `str`
        :param api_key: IBM_API_KEY_ID to access COS
        :type api_key: `str`
        :param service_instance_id: IAM_SERVICE_ID to access COS
        :type service_instance_id: `str`
        :param endpoint: ENDPOINT to access COS
        :type endpoint: `str`
        :param auth_endpoint: IBM_AUTH_ENDPOINT to access COS
        :type auth_endpoint: `str`
        :return: ModelStore
        """
        return ModelStore(COSStorage(bucket,
                                     api_key=api_key,
                                     service_instance_id=service_instance_id,
                                     endpoint=endpoint,
                                     auth_endpoint=auth_endpoint))

    def download(self, prefix, destination):
        """
        Downloads a model from the storage backend

        :param prefix: the source path including filename to retrieve
        :type prefix: `str`
        :param destination: local path to store downloaded model
        :return:
        """
        destination = os.path.abspath(destination)

        return self.storage.retrieve(prefix, destination)

    def load(self, model_metadata, prefix, **kwargs):
        """
        Loads a model from the storage backend into memory

        :param model_metadata: Metadata about the model (i.e. type, and architecture)
        :param prefix:  the source path including filename to retrieve
        :return: `model_wrapper.ModelWrapper`
        """
        with TemporaryDirectory() as tmpdir:
            model_file = self.download(prefix, tmpdir)
            # model_file = self.download(prefix, os.path.join(tmpdir, prefix.split('/')[-1]))
            model_wrapper = get_wrapper(model_metadata, model_file, **kwargs)

            return model_wrapper

    def upload(self, model, destination, saved_model_format=SavedModelFormat.NATIVE):
        """
        Uploads a model to the storage backend

        :param model: Wrapper around model containing metadata and instance
        :type model: `model_wrapper.ModelWrapper`
        :param destination: destination prefix on backend storage
        :type destination: `str`
        :param saved_model_format: file/directory format of saved model
        :type saved_model_format: `util.constants.SavedModelFormat`

        :return: path to uploaded model
        """
        with TemporaryDirectory() as tmpdir:
            model_file = model.save(tmpdir, saved_model_format)

            return self.storage.store(model_file, destination)

    def upload_temporary(self, model, destination, saved_model_format=SavedModelFormat.NATIVE):
        """
        Uploads a model to the storage backend at a temporary location, to be called with 'with' command

        :param model: Wrapper around model containing metadata and instance
        :type model: `model_wrapper.ModelWrapper`
        :param destination: destination prefix on backend storage
        :type destination: `str`
        :param saved_model_format: file/directory format of saved model
        :type saved_model_format: `util.constants.SavedModelFormat`

        :return: path to uploaded model
        """
        with TemporaryDirectory() as tmpdir:
            model_file = model.save(tmpdir, saved_model_format)
            tmp_storage_object = self.storage.store_temporary(model_file, destination)
            return tmp_storage_object

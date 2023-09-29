import os
from ood_enabler.storage.local_storage import FileSystemStorage
from tests.storage.test_storage import StorageTest, json_data_test


class TestLocalStorage(StorageTest):
    def get_test_storage(self):
        return FileSystemStorage()

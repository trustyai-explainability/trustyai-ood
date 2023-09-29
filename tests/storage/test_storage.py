import os
import json
from abc import ABC, abstractmethod
from pytest import fixture
from tempfile import TemporaryDirectory


@fixture
def json_data_test():
    json_data = {'test': 'pytest'}
    json_filename = 'pytest.json'

    return json_data, json_filename


class StorageTest(ABC):
    @abstractmethod
    def get_test_storage(self):
        raise NotImplementedError

    def test_store(self, json_data_test):
        storage = self.get_test_storage()

        with TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, json_data_test[1])
            upload_path = 'pytest'

            with open(fname, 'w') as f:
                json.dump(json_data_test[0], f)

            path = storage.store(fname, upload_path)
            assert path is not None

    def test_retrieve(self, json_data_test):
        storage = self.get_test_storage()

        with TemporaryDirectory() as tmpdir:
            source_path = 'pytest/{}'.format(json_data_test[1])
            file = storage.retrieve(source_path, tmpdir)
            assert os.path.exists(file)

            with open(file, 'r') as f:
                data = json.load(f)

            assert data == json_data_test[0]

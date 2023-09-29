import os
from pytest import fixture
from ood_enabler.storage.cos_storage import COSStorage
from tests.storage.test_storage import StorageTest, json_data_test


@fixture
def bucket():
    return 'rhods'


@fixture
def cos_instance(bucket):
    return COSStorage(bucket=bucket)


class TestCOSStorageTest(StorageTest):
    def get_test_storage(self):
        return COSStorage(bucket='rhods')

    def test_init_cos(self, cos_instance, bucket):
        # ASSUME CREDENTIALS ARE COMING FROM ENV VARIABLES
        assert cos_instance.bucket == bucket
        assert cos_instance.api_key == os.environ['IBM_API_KEY_ID']
        assert cos_instance.service_id == os.environ['IAM_SERVICE_ID']
        assert cos_instance.endpoint == os.environ['ENDPOINT']
        assert cos_instance.auth_endpoint == os.environ['IBM_AUTH_ENDPOINT']

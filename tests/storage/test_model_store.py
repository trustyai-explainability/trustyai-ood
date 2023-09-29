import os
import json
import torch
import torchvision
import pytest
import tensorflow as tf
from pytest import fixture
from tempfile import TemporaryDirectory
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.storage.cos_storage import COSStorage
from ood_enabler.storage.local_storage import FileSystemStorage
from ood_enabler.model_wrapper.pytorch import PytorchWrapper
from ood_enabler.model_wrapper.tf import TFWrapper



@fixture
def bucket():
    return 'rhods'


@fixture
def cos_instance(bucket):
    return ModelStore.from_cos(bucket=bucket)


@fixture
def local_instance():
    return ModelStore.from_filesystem()

@fixture
def pytorch_wrapper():
    model = torchvision.models.resnet50()
    model_metadata = {'type': 'pytorch', 'arch': 'resnet50'}
    return PytorchWrapper(model_metadata, model)

@fixture
def tf_wrapper():
    model = tf.keras.applications.resnet50.ResNet50()
    model_metadata = {'type': 'tf', 'arch': 'resnet50'}
    return TFWrapper(model_metadata, model)


@pytest.mark.parametrize('test_type, expected', [
    ('cos_instance', COSStorage),
    ('local_instance', FileSystemStorage)])
def test_init_store(test_type, expected, request):
    store = request.getfixturevalue(test_type)
    assert type(store.storage) is expected


@pytest.mark.parametrize('test_type', [
    'cos_instance',
    'local_instance'])
@pytest.mark.parametrize('model_type', [
    'pytorch_wrapper',
    'tf_wrapper'])
def test_store_and_retrieve(test_type, model_type, request):
    store = request.getfixturevalue(test_type)
    model = request.getfixturevalue(model_type)

    with TemporaryDirectory() as tmpdir:
        upload_path = 'pytest'
        if type(store.storage) is FileSystemStorage:
            upload_path = os.path.join(tmpdir, upload_path)

        # upload
        path = store.upload(model, upload_path)
        assert path is not None

        #download
        if type(store.storage) is COSStorage:
            p = path.split('/')[1:]
            path = "/".join(p)

        download_path = os.path.join(tmpdir, 'downloads')
        os.makedirs(download_path)
        assert download_path in store.download(path, download_path)


        #load
        wrapper = store.load(model.model_metadata, path)
        assert type(wrapper) == type(model)

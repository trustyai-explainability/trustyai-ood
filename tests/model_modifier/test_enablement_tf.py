import os
import json
import torch
import torchvision
import pytest
import tensorflow as tf
from pytest import fixture
from tempfile import TemporaryDirectory
from ood_enabler.storage.model_store import ModelStore
from tests.helper_test_utils_tf import *

#architectures = ['resnet50', 'vgg16', 'inception_v3', 'efficientnet_v2', 'mobilenet_v2', 'mnasnet1_0']
architectures = ['resnet50']
class Test_enablement_tf:
    test_image = 'test_img.jpg'

    def verify_enablement(self, model, img, data_handler=''):
        passed = 'failed'

        image = Image.open(img)
        image = np.array(image).astype('float32') / 255
        image = transform.resize(image, (224, 224, 3))
        image = np.expand_dims(image, axis=0)
        output = model.model.predict(image)

        if len(output) != 2:
            pass
        elif data_handler != '' and (output[1] < 0 or output[1] > 1):
            pass
        elif not math.isfinite(output[1]):
            pass
        else:
            passed = 'passed'
        assert passed == 'passed'


    @pytest.mark.parametrize("arc", architectures)
    def test_enablment(self, arc):
        model_orig = return_model_handle(arc)
        model_orig = model_preprocessing(model_orig, arc, local_storage='')
        model_ood = model_orig
        OODEnabler.ood_enable(model_ood)
        self.verify_enablement(model_ood, self.test_image)


    @pytest.mark.parametrize("arc", architectures)
    def test_enablment_norm(self, arc):
        data_handler = data_preprocessing(
            dataset_location="https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_small.tar.gz")
        model_orig = return_model_handle(arc)
        model_orig = model_preprocessing(model_orig, arc, local_storage='')
        model_ood_norm = model_orig
        OODEnabler.ood_enable(model_ood_norm, data_handler)
        self.verify_enablement(model_ood_norm, self.test_image, data_handler)

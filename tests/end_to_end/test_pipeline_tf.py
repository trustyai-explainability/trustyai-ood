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

architectures = ['resnet50', 'vgg16', 'mobilenet_v2', 'mnasnet1_0']
#architectures = ['resnet50']

class Test_pipeline_tf:
    test_image = 'test_img.jpg'

    def verify_outputs(self, model, model_enabled, test_dataloader, data_handler=''):

        (c10_x_1, c10_y_1), (c10_x_2, c10_y_2) = tf.keras.datasets.cifar10.load_data()
        nn = tf.image.resize(
            [c10_x_1[0]],
            (224, 224),
            preserve_aspect_ratio=False,
            antialias=False,
            name=None)

        outputs = model.model.predict(nn)
        outputs_enabled = model_enabled.model.predict(nn)

        if len(outputs_enabled) != 2:
            return 'failed'

        o1 = outputs
        o2 = outputs_enabled[0]
        o3 = outputs_enabled[1]

        comp = o1 == o2
        if comp == 'False':
            return 'failed'

        if not math.isfinite(o3.any()):
            return 'failed'

        if data_handler != '' and (o3.any() < 0 or o3.any() > 1):
            return 'failed'

        return 'passed'


    @pytest.mark.parametrize("arc", architectures)
    def test_pipeline_tf(self, arc):
        test_dataloader = ""
        model_orig = return_model_handle(arc)
        model_orig = model_preprocessing(model_orig, arc, local_storage='')
        model_ood = model_orig
        OODEnabler.ood_enable(model_ood)
        assert 'passed' == self.verify_outputs(model_orig, model_ood, test_dataloader)


    @pytest.mark.parametrize("arc", architectures)
    def test_pipeline_norm_tf(self, arc):
        test_dataloader = ""
        data_handler = data_preprocessing(
            dataset_location="https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_small.tar.gz")
        model_orig = return_model_handle(arc)
        model_orig = model_preprocessing(model_orig, arc, local_storage='')
        model_ood_norm = model_orig
        OODEnabler.ood_enable(model_ood_norm, data_handler)
        assert 'passed' == self.verify_outputs(model_orig, model_ood_norm, test_dataloader, data_handler)

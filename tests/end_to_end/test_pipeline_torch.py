import os
import json
import torch
import torchvision
import pytest
import tensorflow as tf
from pytest import fixture
from tempfile import TemporaryDirectory
from ood_enabler.storage.model_store import ModelStore
from tests.helper_test_utils import *

#architectures = ['resnet50', 'vgg16', 'inception_v3', 'efficientnet_v2', 'mobilenet_v2', 'mnasnet1_0']
architectures = ['resnet50']
class Test_pipeline_torch:
    test_image = 'test_img.jpg'

    def verify_outputs(self, model, model_enabled, test_dataloader, data_handler=''):
        model.model.eval()
        model_enabled.model.eval()

        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            outputs_orig = model.model(inputs)
            outputs_enabled = model_enabled.model(inputs)

            if len(outputs_enabled) != 2:
                return 'failed'

            o1 = outputs_orig.detach().numpy()
            o2 = outputs_enabled[0].detach().numpy()
            o3 = outputs_enabled[1].detach().numpy()

            comp = o1 == o2
            if comp.any() == 'False':
                return 'failed'

            if not math.isfinite(o3.any()):
                return 'failed'

            if data_handler != '' and (o3.any() < 0 or o3.any() > 1):
                return 'failed'

        return 'passed'

    @pytest.mark.parametrize("arc", architectures)
    def test_pipeline(self, arc):
        test_dataloader = helper_data_preprocessing(test_dataset="")
        model_orig = return_model_handle(arc)
        model_orig = model_preprocessing(model_orig, arc, local_storage='')
        model_ood = copy.deepcopy(model_orig)
        OODEnabler.ood_enable(model_ood)
        assert 'passed' == self.verify_outputs(model_orig, model_ood, test_dataloader)


    @pytest.mark.parametrize("arc", architectures)
    def test_pipeline_norm(self, arc):
        test_dataloader = helper_data_preprocessing(test_dataset="")
        data_handler = data_preprocessing(
            dataset_location="https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_small.tar.gz")
        model_orig = return_model_handle(arc)
        model_orig = model_preprocessing(model_orig, arc, local_storage='')
        model_ood_norm = copy.deepcopy(model_orig)
        OODEnabler.ood_enable(model_ood_norm, data_handler)
        assert 'passed' == self.verify_outputs(model_orig, model_ood_norm, test_dataloader, data_handler)

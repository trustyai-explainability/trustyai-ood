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

#architectures = ['resnet50']
#architectures = ['resnet18', 'resnet50', 'squeezenet1_0', 'vgg16', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet1_0']
#architectures = ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']

architectures = ['swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b']

class Test_enablement_torch:
    test_image = 'test_img.jpg'

    def verify_enablement(self, model, img, data_handler=''):

        passed = 'failed'

        model.model.eval()

        image = Image.open(img)
        image = transform(image).float()
        image = image.unsqueeze(0)

        output = model.model(image)
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
        model_ood = copy.deepcopy(model_orig)
        OODEnabler.ood_enable(model_ood)
        self.verify_enablement(model_ood, self.test_image)


    @pytest.mark.parametrize("arc", architectures)
    def test_enablment_norm(self, arc):
        data_handler = data_preprocessing(
            dataset_location="https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_small.tar.gz")
        model_orig = return_model_handle(arc)
        model_orig = model_preprocessing(model_orig, arc, local_storage='')
        model_ood_norm = copy.deepcopy(model_orig)
        OODEnabler.ood_enable(model_ood_norm, data_handler)
        self.verify_enablement(model_ood_norm, self.test_image, data_handler)
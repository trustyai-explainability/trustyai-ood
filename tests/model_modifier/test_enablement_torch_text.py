import os
import json
import torch
import torchvision
import sys
import pytest
import tensorflow as tf
from pytest import fixture
from tempfile import TemporaryDirectory

# from ood_enabler.ood_enabler import OODEnabler
# from ood_enabler.storage.model_store import ModelStore
from tests.helper_test_utils import *
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

architectures = ['roberta']

class Test_text_model_enablement_torch:
   
    def verify_enablement(self, model, dataloader, data_handler=''):
        passed = 'failed'
        print(dataloader)
        for i, data in enumerate(dataloader):
            data_point = data
            break 
        model.model.eval()
        output = model.model(data_point["input_ids"], data_point["attention_mask"])
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
        model_metadata = {'type': 'pytorch', 'arch': arc} 
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        model = PytorchWrapper(model_metadata, model=model_orig)
        OODEnabler.ood_enable(model)
        dataloader = helper_text_data_preprocessing("tweet_eval", MODEL, subset="sentiment")
        self.verify_enablement(model, dataloader)


    @pytest.mark.parametrize("arc", architectures)
    def test_enablment_norm(self, arc):
        model_orig = return_model_handle(arc)
        model_metadata = {'type': 'pytorch', 'arch': arc} 
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        ds_metadata = {'hf_dataset': ("tweet_eval", "sentiment",
                                       "test[:1%]"), 'col_name_to_tokenize': "text", 
                                       'batch_size': 32, 'hf_tokenizer': MODEL }
        model = PytorchWrapper(model_metadata, model=model_orig)
        data_handler = helper_get_text_handler(ds_metadata)
        OODEnabler.ood_enable(model, data_handler)
        dataloader = helper_text_data_preprocessing("tweet_eval", MODEL, subset="sentiment")
        self.verify_enablement(model, dataloader, data_handler)

  

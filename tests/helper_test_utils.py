###imports
import os
import sys
import torch
import torchvision
from tensorflow.keras.utils import get_file
from tempfile import TemporaryDirectory
from ood_enabler.ood_enabler import OODEnabler
from ood_enabler.model_wrapper.pytorch import PytorchWrapper
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.storage.local_storage import FileSystemStorage
from ood_enabler.data.pytorch_image_data_handler import PytorchImageDataHandler
from ood_enabler.data.pytorch_text_data_handler import PytorchTextDataHandler
import time
import copy
from torchvision.datasets import SVHN
from PIL import *
import logging
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from torchvision import transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def return_model_handle(arc):
     if arc == 'resnet18':
             model = torchvision.models.resnet18(pretrained=True)
     elif arc == 'resnet50':
             model = torchvision.models.resnet50(pretrained=True)
     elif arc == 'squeezenet1_0':
             model = torchvision.models.squeezenet1_0(pretrained=True)
     elif arc == 'vgg16':
             model = torchvision.models.vgg16(pretrained=True)
     elif arc == 'densenet161':
             model = torchvision.models.densenet161(pretrained=True)
     elif arc == 'inception_v3':
             model = torchvision.models.inception_v3(pretrained=True)
     elif arc == 'googlenet':
             model = torchvision.models.googlenet(pretrained=True)
     elif arc == 'shufflenet_v2_x1_0':
             model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
     elif arc == 'mobilenet_v2':
             model = torchvision.models.mobilenet_v2(pretrained=True)
     elif arc == 'resnext50_32x4d':
             model = torchvision.models.resnext50_32x4d(pretrained=True)
     elif arc == 'wide_resnet50_2':
             model = torchvision.models.wide_resnet50_2(pretrained=True)
     elif arc == 'mnasnet1_0':
             model = torchvision.models.mnasnet1_0(pretrained=True)
     elif arc == 'vit_b_16':
             model = torchvision.models.vit_b_16(pretrained=True)
     elif arc == 'vit_b_32':
             model = torchvision.models.vit_b_32(pretrained=True)
     elif arc == 'vit_l_16':
             model = torchvision.models.vit_l_16(pretrained=True)
     elif arc == 'vit_l_32':
             model = torchvision.models.vit_l_32(pretrained=True)
     elif arc == 'vit_h_14':
             model = torchvision.models.vit_h_14(pretrained=True)
     elif arc == 'swin_t':
             model = torchvision.models.swin_t(pretrained=True)
     elif arc == 'swin_s':
             model = torchvision.models.swin_s(pretrained=True)
     elif arc == 'swin_b':
             model = torchvision.models.swin_b(pretrained=True)
     elif arc == 'swin_v2_t':
             model = torchvision.models.swin_v2_t(pretrained=True)
     elif arc == 'swin_v2_s':
             model = torchvision.models.swin_v2_s(pretrained=True)
     elif arc == 'swin_v2_b':
             model = torchvision.models.swin_v2_b(pretrained=True) 
     elif arc == 'roberta':
             MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
             model = AutoModelForSequenceClassification.from_pretrained(MODEL)


     return model



def data_preprocessing(dataset_location):

     if 'https' in dataset_location:
        #load from COS
        archive = get_file(origin=dataset_location, extract=False, cache_dir='./tmp')
        local_store = FileSystemStorage()
        ds_metadata = {'img_height': 224, 'img_width': 224, 'batch_size': 32, 'normalize': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
        #data_handler = PytorchImageDataHandler(PytorchImageDataHandler.load_dataset(local_store, archive, '.', ds_metadata))
        data_handler = PytorchImageDataHandler()
        data_handler.load_dataset(local_store, archive, '.', ds_metadata)
     else:
        #get data handler from local storage
        pass

     return data_handler



def model_preprocessing(model, arc, local_storage=''):

     model_store = ModelStore.from_filesystem()
     model_metadata = {'type': 'pytorch', 'arch': arc}

     with TemporaryDirectory() as tmpdir:
       model_path = os.path.join(tmpdir, 'pytorch_'+arc)
       model_full = torch.jit.script(model)
       model_full.save(model_path)

       model = model_store.load(model_metadata, model_path)

     return model
 
 

def helper_data_preprocessing(test_dataset=""):

     batch_size = 32

     #dataset = torchvision.datasets.SVHN(root='./data', split='train',
     #                                    download=True, transform=transform)
     #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


     dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

     #take a subsample of this dataset
     return dataloader

def helper_preprocess_tweet(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def helper_text_data_preprocessing(dataset, tokenizer_path, col_name_to_tokenize="text", subset=None, batch_size=1, split="train"):
        if subset is not None:
               dataset = load_dataset(dataset, subset, split=split)
        else:
               dataset = load_dataset(dataset, split=split)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=512)
        if dataset == "tweet_eval":
                for text in dataset["text"]:
                        text = helper_preprocess_tweet(text)
        dataset = dataset.map(lambda e: tokenizer(e[col_name_to_tokenize], truncation=True, padding='max_length') , batched=True)
        dataset = dataset.remove_columns(list(filter(lambda x: x not in ["attention_mask", "input_ids"],  dataset.features)))
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
        return dataloader

def helper_get_text_handler(ds_metadata):
        # dummy url to get an archive to pass into the data handler
        dataset_url = "https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_small.tar.gz"
        archive = get_file(origin=dataset_url, extract=False)
        local_store = FileSystemStorage()
        data_handler = PytorchTextDataHandler()
        data_handler.load_dataset(local_store, archive, '.', ds_metadata)
        return data_handler


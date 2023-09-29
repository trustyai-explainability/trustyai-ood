###imports
import os
import sys
import tensorflow as tf
from tempfile import TemporaryDirectory

#from tensorflow.keras.utils import get_file

from ood_enabler.ood_enabler import OODEnabler
from ood_enabler.model_wrapper.tf import TFWrapper
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.storage.local_storage import FileSystemStorage
from ood_enabler.data.tf_image_data_handler import TFImageDataHandler

import time
import copy
from PIL import *
import logging
import math
import numpy as np

from skimage import transform


def return_model_handle(arc):


     if arc == 'resnet50':
             model = tf.keras.applications.resnet50.ResNet50()
     elif arc == 'vgg16':
             model = tf.keras.applications.vgg16.VGG16()
     elif arc == 'densenet121':
             model = tf.keras.applications.densenet.DenseNet121()
     elif arc == 'inception_v3':
             model = tf.keras.applications.inception_v3.InceptionV3()
     elif arc == 'mobilenet_v2':
             model = tf.keras.applications.mobilenet_v2.MobileNetV2()
     elif arc == 'mnasnet1_0':
             model = tf.keras.applications.nasnet.NASNetMobile()
     elif arc == 'efficientnet_v2':
             model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0()

     return model



def data_preprocessing(dataset_location):

     if 'https' in dataset_location:
        #load from COS
        archive = tf.keras.utils.get_file(origin=dataset_location, extract=False)
        local_store = FileSystemStorage()
        ds_metadata = {'img_height': 224, 'img_width': 224, 'batch_size': 32, 'normalize': 255}
        data_handler = TFImageDataHandler()
        data_handler.load_dataset(local_store, archive, '.', ds_metadata)
     else:
        #get data handler from local storage
        pass

     return data_handler



def model_preprocessing(model, arc, local_storage=''):

     model_store = ModelStore.from_filesystem()
     model_metadata = {'type': 'tf', 'arch': arc}

     with TemporaryDirectory() as tmpdir:
       model_path = os.path.join(tmpdir, 'tf_'+arc)
       model.save(model_path)

       model = model_store.load(model_metadata, model_path)

     return model
 
 

'''
def test_data_preprocessing(test_dataset=""):

     batch_size = 32

     #dataset = torchvision.datasets.SVHN(root='./data', split='train',
     #                                    download=True, transform=transform)
     #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


     dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

     #take a subsample of this dataset
     return dataloader
'''

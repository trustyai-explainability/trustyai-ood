import os
import sys
from tensorflow.keras.utils import get_file
from tempfile import TemporaryDirectory
import datetime
import tensorflow as tf

ood_path = os.path.abspath('../')
if ood_path not in sys.path:
    sys.path.append(ood_path)

from ood_enabler.util.constants import SavedModelFormat
from ood_enabler.ood_enabler import OODEnabler
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.storage.local_storage import FileSystemStorage
from ood_enabler.data.tf_image_data_handler import TFImageDataHandler
from ood_enabler.util.archiver import archive
from ood_enabler.util.constants import InferenceServiceType

dataset_url = "https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_very_small.tar.gz"
archive_path = get_file(origin=dataset_url, extract=False)

local_store = FileSystemStorage()
ds_metadata = {'img_height': 224, 'img_width': 224, 'batch_size': 8, 'normalize': 255}

data_handler = TFImageDataHandler()
data_handler.load_dataset(local_store, archive_path, '.', ds_metadata)

model_store = ModelStore.from_filesystem()
model_store_cos = ModelStore.from_cos('rhods')

model = tf.keras.applications.resnet50.ResNet50()
model_metadata = {'type': 'tf', 'arch': 'resnet50', 'ood_thresh_percentile': 20}

with TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, 'tf_resnet50')
    model.save(model_path)
    archive_path = archive(model_path, tmpdir)

    model = model_store.load(model_metadata, archive_path)

OODEnabler.ood_enable(model, data_handler, inference_service=InferenceServiceType.RHODS, bucket='rhods')
# OODEnabler.ood_enable(model)

(c10_x_1, c10_y_1), (c10_x_2, c10_y_2) = tf.keras.datasets.cifar10.load_data()
input = tf.image.resize(
    [c10_x_1[0]],
    (224, 224),
    preserve_aspect_ratio=False,
    antialias=False,
    name=None)

o2 = model.model.predict(input)

print(o2)

timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))

path = model_store_cos.upload(model, 'model_test_{}'.format(timestamp), saved_model_format=SavedModelFormat.NATIVE)

print(path)

model_new = model_store_cos.load(model_metadata, 'model_test_{}'.format(timestamp), bypass_model_check=True)

outputs_new = model_new.model.predict(input)

print(outputs_new)

import os
import sys
import torch
import torchvision
from tensorflow.keras.utils import get_file
from tempfile import TemporaryDirectory
from torchvision import transforms
import datetime

ood_path = os.path.abspath('../')
if ood_path not in sys.path:
    sys.path.append(ood_path)

from ood_enabler.util.constants import SavedModelFormat
from ood_enabler.ood_enabler import OODEnabler
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.storage.local_storage import FileSystemStorage
from ood_enabler.data.pytorch_image_data_handler import PytorchImageDataHandler

dataset_url = "https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_small.tar.gz"
archive = get_file(origin=dataset_url, extract=False)

local_store = FileSystemStorage()
ds_metadata = {'img_height': 224, 'img_width': 224, 'batch_size': 32, 'normalize': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

data_handler = PytorchImageDataHandler()
data_handler.load_dataset(local_store, archive, '.', ds_metadata)

model_store = ModelStore.from_filesystem()
# model_store_cos = ModelStore.from_cos('rhods')

model = torchvision.models.resnet50(pretrained=True)
model_metadata = {'type': 'pytorch', 'arch': 'resnet50', 'ood_thresh_percentile': 20}

with TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, 'pytorch_resnet50')
    model_full = torch.jit.script(model)
    model_full.save(model_path)

    model = model_store.load(model_metadata, model_path)

OODEnabler.ood_enable(model, data_handler)
# OODEnabler.ood_enable(model)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

# OOD model
model.model.eval()
for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    outputs = model.model(inputs)
    break

print(outputs)

# timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))

# path = model_store_cos.upload(model, 'model_test_{}'.format(timestamp), saved_model_format=SavedModelFormat.TORCHSERVE)
# # path = model_store_cos.upload(model, 'test_model_single_file.pt')
# print(path)
# model_new = model_store_cos.load(model_metadata, 'model_test_{}'.format(timestamp), bypass_model_check=True)
# # model_new = model_store_cos.load(model_metadata, 'test_model_single_file.pt')

# outputs_new = model_new.model(inputs)

# print(outputs_new)

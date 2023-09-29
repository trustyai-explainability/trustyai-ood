import os
import sys
import torch
import torchvision
from tensorflow.keras.utils import get_file
from torchvision import transforms

ood_path = os.path.abspath('../')
if ood_path not in sys.path:
    sys.path.append(ood_path)

from ood_enabler.ood_enabler import OODEnabler
from ood_enabler.model_wrapper.pytorch import PytorchWrapper
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.storage.local_storage import FileSystemStorage
from ood_enabler.data.pytorch_image_data_handler import PytorchImageDataHandler

def main():
    print("YoloV5 inferencing batch of images with datahandler\n")
    dataset_url = "https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_small.tar.gz"
    archive = get_file(origin=dataset_url, extract=False)
    local_store = FileSystemStorage()
    ds_metadata = {'img_height': 224, 'img_width': 224, 'batch_size': 32,
                   'normalize': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
    data_handler = PytorchImageDataHandler()
    data_handler.load_dataset(local_store, archive, '.', ds_metadata)
    model_store = ModelStore.from_filesystem()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    model_metadata = {'type': 'pytorch', 'arch': 'yolov5l', 'ood_thresh_percentile': 20}
    model = PytorchWrapper(model_metadata, model=model)
    OODEnabler.ood_enable(model, data_handler)
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
        inputs, labels = data
        print("inputs.shape:{}".format(inputs.shape))
        outputs = model.model(inputs)
        print("type(outputs):{}".format(type(outputs)))
        print(outputs)
        break

if __name__ == '__main__':
    main()



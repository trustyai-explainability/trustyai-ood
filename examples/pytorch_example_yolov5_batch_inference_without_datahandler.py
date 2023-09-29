import os
import sys
import torch
import torchvision
from torchvision import transforms

ood_path = os.path.abspath('../')
if ood_path not in sys.path:
    sys.path.append(ood_path)

from ood_enabler.ood_enabler import OODEnabler
from ood_enabler.model_wrapper.pytorch import PytorchWrapper

def main():
    print("YoloV5 inferencing batch of images without datahandler\n")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    model_metadata = {'type': 'pytorch', 'arch': 'yolov5l', 'ood_thresh_percentile': 20}
    model = PytorchWrapper(model_metadata, model=model)
    OODEnabler.ood_enable(model)
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



import torchvision
import transformers
from transformers import AutoModelForSequenceClassification


def get_pytorch_base_model(arch):
    if arch == 'resnet18':
        return torchvision.models.resnet18(pretrained=True)
    elif arch == 'resnet50':
        print('returning this')
        return torchvision.models.resnet50(pretrained=True)
    elif arch == 'squeezenet1_0':
        return torchvision.models.squeezenet1_0(pretrained=True)
    elif arch == 'vgg16':
        return torchvision.models.vgg16(pretrained=True)
    elif arch == 'densenet161':
        return torchvision.models.densenet161(pretrained=True)
    elif arch == 'shufflenet_v2_x1_0':
        return torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif arch == 'mobilenet_v2':
        return torchvision.models.mobilenet_v2(pretrained=True)
    elif arch == 'resnext50_32x4d':
        return torchvision.models.resnext50_32x4d(pretrained=True)
    elif arch == 'wide_resnet50_2':
        return torchvision.models.wide_resnet50_2(pretrained=True)
    elif arch == 'mnasnet1_0':
        return torchvision.models.mnasnet1_0(pretrained=True)
    elif arch == 'roberta':
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        return AutoModelForSequenceClassification.from_pretrained(MODEL, torchscript=True)
    else:
        raise ValueError('unsupported model arch provided')

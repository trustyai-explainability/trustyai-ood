import os
import sys
import torch
import torchvision
import transformers
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
from ood_enabler.data.pytorch_text_data_handler import PytorchTextDataHandler
from ood_enabler.model_wrapper.pytorch import PytorchWrapper

dataset = load_dataset("tweet_eval", "sentiment", split='test')


dataset_url = "https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/flower_photos_small.tar.gz"
archive = get_file(origin=dataset_url, extract=False)

local_store = FileSystemStorage()
ds_metadata = {'hf_dataset': ("tweet_eval", "sentiment", "test"), 'col_name_to_tokenize': "text", 'batch_size': 32, 'hf_tokenizer': "bert-base-cased" }

data_handler = PytorchTextDataHandler()
data_handler.load_dataset(local_store, archive, '.', ds_metadata)

model_store = ModelStore.from_filesystem()
model_store_cos = ModelStore.from_cos('rhods')

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
pretrained_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model_metadata = {'type': 'pytorch', 'arch': 'roberta', 'ood_thresh_percentile': 20}

model = PytorchWrapper(model_metadata, model=pretrained_model)

OODEnabler.ood_enable(model)
# OODEnabler.ood_enable(model, data_handler)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

batch_size = 32

trainset = load_dataset("tweet_eval", "sentiment", split='train')
print(trainset)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
for text in trainset["text"]:
    text = preprocess(text)

trainset = trainset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length') , batched=True)
trainset = trainset.remove_columns(list(filter(lambda x: x not in ["attention_mask", "input_ids"],  trainset.features)))
trainset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size)
# OOD model
model.model.eval()
for i, data in enumerate(trainloader):
    # get the inputs; data is a list of [inputs, labels]
    outputs = model.model(data["input_ids"], data["attention_mask"])
    print(outputs)
    break
print("Printing the final outputs in the example")
print(outputs)

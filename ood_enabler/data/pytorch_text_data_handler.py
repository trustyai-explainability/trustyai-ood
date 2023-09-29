import os
import torch
import transformers
import datasets
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from ood_enabler.data.data_handler import DataHandler
from ood_enabler.exceptions.exceptions import OODEnableException
from ood_enabler.util.archiver import extract_archive

def preprocess_tweet(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class PytorchTextDataHandler(DataHandler):
    """
    Class for loading and preprocessing image dataset for Pytorch backend
    """
    def __init__(self, data=None):
        """
        Intializes datahander with in-memory dataset
        :param dataset: image dataset loaded into memory
        :type dataset: `torch.Tensor` or `torch.util.data.DataLoader`
        """
        super().__init__(data)

    def _load_dataset(self, storage, source, destination, metadata):
        """
        Downloads and loads a dataset from storage backend

        
        :param metadata: info about the text data set hf_dataset 
                        "hf_dataset": [data set name (can be csv or HF dataset name), subset (can be None), split (can be None)] :type list,
                        "hf_tokenizer"": name of tokenizer to use :type string, 
                        "col_name_to_tokenize": name of column within the dataset to be tokenized :type string,
                        "batch_size": default is 32 :type int

        :type metadata: `dict`
        :return: loaded dataset
        :rtype pytorch DataLoader with batched text data 
        """
        dataset_params = metadata["hf_dataset"] # expect a list here of format [dataset, subset, split]

        tokenizer = AutoTokenizer.from_pretrained(metadata["hf_tokenizer"],  model_max_length=512)
        dataset = []
        if dataset_params[2] is None: 
            # default split is always train
            dataset_params[2] = "train"
        if dataset_params[1] is None: 
            if ".csv" in dataset_params[0]:
                df = pd.read_csv(dataset_params[0])
                df = pd.DataFrame(df)
                dataset = Dataset.from_pandas(df, split= dataset_params[2])
            else: 
                # load huggingface dataset with no subset
                dataset = load_dataset(dataset_params[0], split=dataset_params[2])
        else: 
            # load huggingface dataset with subset specified
            dataset = load_dataset(dataset_params[0], dataset_params[1], split=dataset_params[2])
        if dataset_params[0] == "tweet_eval":
            for text in dataset[metadata["col_name_to_tokenize"]]:
                text = preprocess_tweet(text)
        dataset = dataset.map(lambda e: tokenizer(e[(metadata["col_name_to_tokenize"])], 
                                                  truncation=True, 
                                                  padding='max_length'), 
                                                  batched=True)
        dataset = dataset.remove_columns(list(filter(lambda x: x not in ["attention_mask", "input_ids"],  dataset.features)))
        print(dataset)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=metadata.get('batch_size', 32))
        return dataloader

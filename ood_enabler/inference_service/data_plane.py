"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
import requests
import json
import base64
import numpy as np


def remote_infer(model, data_handler, url, headers=None):
    """
    Performs inference remotely via url/header, following the V2 inference protocol of KServe

    :param model: model to use for inference
    :type model: `model.ModelWrapper`
    :param data_handler: data to be inferred
    :type data_handler: `data.Datahandler`
    :return: stats for normalizing data
    :param url: URL of inference service request
    :type url: string
    :param headers: header of inference service request (optional)
    :type headers: dict
    """

    def func(data_batch):
        if isinstance(data_batch, np.ndarray):
            data_batch_np = data_batch
        else:
            try:
                data_batch_np = data_batch.numpy()
            except:
                try:
                    data_batch_np = np.array(data_batch)
                except:
                    raise ValueError('Unsupported data type for data_batch')

        # image_64_encode = base64.b64encode(data_batch_np.tobytes())
        # bytes_array = image_64_encode.decode('utf-8')
        request_data = {'inputs': [
            {
                'name': 'input',
                'shape': list(data_batch_np.shape),
                'datatype': 'FP32',
                # 'parameters': {'content_type': 'base64'},
                # 'data': [bytes_array]
                'data': data_batch_np.tolist()
            }
        ]}
        if headers is not None:
            r = requests.post(url, headers=headers, data=json.dumps(request_data))
        else:
            r = requests.post(url, data=json.dumps(request_data))

        if r.status_code != 200:
            raise Exception(r.text)

        response_dict = json.loads(r.content)
        outputs = response_dict['outputs']
        batch_size = data_batch_np.shape[0]
        logits = None
        ood_scores = None
        for o in outputs:
            if o['name'] == 'predictions' or o['name'] == 'logits':
                logits = np.array(o['data']).reshape([batch_size, -1])
            elif o['name'] == 'ood_scores':
                ood_scores = np.array(o['data']).reshape([batch_size, 1])
        return logits, ood_scores

    inference_results = model.infer(data_handler, func)

    return inference_results

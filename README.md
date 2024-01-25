# Library for Model Certainty Enablement with Out-of-Distribution (OOD) Detection
## Table of Contents
- [Library Description](#api)
    - [Supported ML Frameworks and Model Architectures](#framew)
    - [Supported Inference Backends for Normalized Certainty Enablement](#backend)
- [Installation and Quickstart](#install)
- [Running Certainty Enablement as a Container](#container)
- [Using a Certainty Enabled Model in Openshift Rhods](#rhods)

## Library Description <a name="api"></a>
This library offers the capability to enable a user-provided deep learning model with a certainty prediction layer that can be used to detect out-of-distribution data samples. In the current implementation, to get a certainty enabled model, the library will take in the original model and optionally an in-distribution (normal) dataset, then the output is a modified model which will be stored at a user-specified location. The modified model is capable of generating the original model inference output and a certainty score. (*low* for out-of-distribution samples and *high* for in-distribution samples) at inference time.
The key classes and constructs of the library and the interactions between them are
shown in the figure below.

<html>
<div style="text-align: center;">
<img src="https://media.github.ibm.com/user/16743/files/f5d57d34-2258-4193-ba42-661d2e529e74" width="100%">
</div>
</html>


- **Data Handler**: This class handles data loading and processing to prepare the data to be used by downstream ML frameworks.
The abstract parent class has child sub-classes for PyTorch and TensorFlow ML backends. The key methods here are
````load data```` and ``process data``. ````load data```` is called to load data from a local file storage
or a COS location specified as an argument.``process data`` is called to prepare ML backend specific data loaders.

- **Storage**: This class handles the interactions between the library and local filesystem or a COS location in order to
retrieve and store datasets and models. This class
is used by both `Data Handler` and `Model Store` classes. The abstract parent class has child sub-classes
for PyTorch and TensorFlow ML backends. The key methods here are
``retrieve`` and `store`. ``retrieve`` is called to get data or a model from a local file storage or a COS
location.``store`` is called to store data or a model in a local file storage or a COS location.

- **Model Store**: This class provides an interface between `Model Wrapper` and `Storage` classes addressing model retrieval and storage to local filesystem or COS location.

- **Model Wrapper**: This class creates a wrapper abstraction around a user-provided model for smooth downstream model processing and certainty
enablement. It additionally performs correctness checks on user-provided model files.
The abstract parent class has child sub-classes for PyTorch, TensorFlow and ONNX ML backends. The key methods here are
``load``, `save`, and `infer`. ``load`` and `save` provide I/O interactions for an input model for PyTorch, TensorFlow and ONNX
backends.``infer`` calls the *model native* inference method in order to forward pass data through the model
to obtain inference results (including a *certainty score*).

- **Inference Service**: This class provides an interface between the library and different inference backends. Currently, the
library supports ``in memory`` and `KServe` inference backends. The key method here is `infer` for
performing model inference. In addition to this, for ``KServe`` backend,
the class provides methods to connect and deploy a model in ``KServe``.


- **Model Modifier**: This class provides methods to add additional layers to an input model which can produce a certainty score
(in addition to regular model inference output) as well as a normalized certainty score (if a user provides an *input dataset*).
The two key methods are ``add_ood_layer`` and `add_normalization`. `add_ood_layer` takes in an input model and adds a layer that can produce
a certainty score at inference time. `add_normalization`- If a user provides an *in-distribution dataset*, this method additionally adds a *normalization layer* into the
certainty enabled model ensuring that the certainty scores will always be between **0** and **1**.

- **OOD Enabler**: This class has the ``OOD_enable`` method which is the primary interface of the libary interacting with all the classes.
The inputs to ``OOD_enable`` are `model wrapper` and `data handler` objects, and an `inference service` argument
specifying the user-preferred inference service.


### Supported ML Frameworks and Model Architectures <a name="framew"></a>
Currently, we support deep learning model format as input. We follow the convention of deep learning model architectures from [Torchvision](https://pytorch.org/vision/stable/models.html) and from the [TensorFlow/Keras model zoo](https://www.tensorflow.org/api_docs/python/tf/keras/applications) pages. We are in the process of adding more architectural support.

The following are model architectures which we have verified that our library supports.  

**PyTorch:** Resnet18/50, Mobilenet v2, Resnext50, Wide Resnet50, Mnasnet, Vision Transformer (ViT), Swin-Transformer (swin), Yolo v5

**Tensorflow:** Resnet18/50, VGG16, Mobilenet v2, Mnasnet

**ONNX:** Resnet18/50

**Huggingface:** All encoder only classification models that implement HuggingFace's
`transformers.AutoModelForSequenceClassification` [library](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification) (e.g., DistilBert, Albert, Camembert, XLMRoberta, Roberta, Bert, XLNet, XLM, Flaubert)

More information on the support of different architecture and inference backend combinations is given [here](https://github.ibm.com/Distributed-Data/ood/blob/main/ood_enabler/inference_service/README.md#supported-architectures-and-backends).


### Supported Inference Backends for Normalized Certainty Enablement <a name="backend"></a>
Currently the library supports two ways of certainty enablement (with and without normalization). For normalization, in addition to user-provided model, an in-distribution dataset is required. This is needed as the normalization step involves obtaining the distribution of certainty scores on the in-distribution dataset.
This means as an internal step, we will pass the in-distribution dataset into the model for inference. The inference can be performed using in-memory native ML backend (CPU/GPU), KServe, or RHODS (modelmesh). For more details on how to use the KServe and RHODS inference backends with this library, please see [here](https://github.ibm.com/Distributed-Data/ood/blob/main/ood_enabler/inference_service/README.md#descriptions-and-setup-instructions).

The following table shows the different testing environments of the certainty enablement framework, with different settings of the inference backend.

| Certainty Enablement Environment | Without External Inference Backend| KServe as Inference Backend | RHODS as Inference Backend |
|---------------|---------| --- |-----------------|
| Local Python environment  | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark:  |
| K8s   | being tested | being tested | --- |
| Openshift | being tested | --- |  being tested  |


## Installation and Quickstart <a name="install"></a>
This repo can be cloned using

```
git clone https://github.ibm.com/Distributed-Data/ood.git
```

A conda environment or python virtual environment is recommended for using the library in order to not conflict with other libraries
or settings on the system. Requirements should be installed using

```
pip install -r requirements.txt
```


### Steps to Run the Certainty enablement Pipeline
We will use the PyTorch ML backend as an example to walk through the steps needed to run the certainty enablement pipeline (steps using TensorFlow backend are quite identical and can be seen in these [notebooks and python examples](#examples)).



#### Create Model Store connection to COS and get user provided model
A user can provide their own model (from among [supported architectures](#framew)) stored at a COS storage location with corresponding credentials.

```bash
credentials = {
  "apikey": "<YOUR_API_KEY>",
  "resource_instance_id": "<YOUR_RESOURCE_INSTANCE_ID>"
}

bucket = '<YOUR_BUCKET>'
service_endpoint = 'https://s3.us-east.cloud-object-storage.appdomain.cloud'
auth_endpoint = 'https://iam.cloud.ibm.com/oidc/token'

```

With the provided credentials, a ``Model Store`` connection to COS is established and the model is loaded as shown in steps below:

```bash
model_metadata = {'type': 'pytorch', 'arch': 'resnet50'}

model_store = ModelStore.from_cos(bucket,
                                  api_key=credentials['apikey'],
                                  service_instance_id=credentials['resource_instance_id'],
                                  endpoint=service_endpoint,
                                  auth_endpoint=auth_endpoint)



cos_model_path = '<valid model path relative to service endpoint>'
model = model_store.load(model_metadata, cos_model_path)

```

#### Create a data handler from COS
A user can optionally provide an in-distribution dataset (e.g., via a COS location as above). User can additionally provide specific metadata for the dataset such as image size *(height, width)* that is suitable as an input to the model via **ds_metadata** as shown in the example.

```bash
ds_metadata = {'img_height': 224, 'img_width': 224, 'batch_size': 32, 'normalize': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

data_cos = COSStorage(bucket,
                      api_key=credentials['apikey'],
                      service_instance_id=credentials['resource_instance_id'],
                      endpoint=service_endpoint,
                      auth_endpoint=auth_endpoint)


cos_data_path = '<valid data path relative to service endpoint>'

data_handler = get_image_data_handler('pytorch', ds_metadata, data_cos, cos_data_path, '.')
```


#### Certainty enablement without user-provided in-distribution data
When certainty enablement is done without a *data_handler*, it is important to understand that the model, when deployed for inference, will provide un-normalized certainty scores (values will not be between 0 and 1).

```bash
OODEnabler.ood_enable(model)
```

#### Certainty enablement with user-provided in-distribution data
Certainty enablement can be done with an in-distribution training datahandler that will additionally add a normalization layer to the provided model. This model when deployed for inference will provide normalized certainty scores (i.e., value of certainty score will be between 0 and 1). Here, a score of **1** indicates that the model is *highly certain* about the prediction while a score of **0** indictes that the model is *highly uncertain* about the prediction.

```bash
OODEnabler.ood_enable(model,  data_handler)
```

#### Saving certainty enabled model to COS
Finally the certainty enabled (and optionally normalized) model is stored to the user-provided COS location.

```bash
cos_enabled_model_path = '<valid model path relative to service endpoint>'
model_store.upload(model, cos_enabled_model_path)

```

### Examples <a name="examples"></a>
We have created several Python examples showing how to run the library with different configurations using different ML backends.

```bash
cd examples/
examples
├── pytorch_example_resnet50_inference.ipynb
├── pytorch_example_resnet50_inmemory.py
├── pytorch_example_resnet50_kserve.py
├── pytorch_example_resnet50_rhods.py
├── tf_example_resnet50_inference.ipynb
├── tf_example_resnet50_inmemory.py
├── tf_example_resnet50_kserve.py
├── tf_example_resnet50_rhods.py
```

Note that the `.py` examples include uploading and downloading the model to/from COS, with the credentials saved in environment variables. These environment variables include `IBM_API_KEY_ID`, `IAM_SERVICE_ID`, `ENDPOINT`, and `IBM_AUTH_ENDPOINT`. In addition, the COS bucket name needs to be specified in the code (its current value is `'rhods'`).

## Running Certainty enablement as a Container <a name="container"></a>
The library can also be installed as a container with Docker, which supports two options for containers:
  - [REST Service](Docker/Dockerfile.rest)
  - [Ephemeral container (short lived)](Docker/Dockerfile.ephemeral)

### Building Docker Images
#### REST Service
To build REST service container, run the command:
```
docker build . -f Docker/Dockerfile.rest -t ood:latest
```
Documentation on how to deploy the REST Service container and interact with the service via REST APIs can be found [here](ood_enabler/apis/README.md)<br>


#### Ephemeral Container
To build Ephemeral container, run the command:
```
docker build . -f Docker/Dockerfile.ephemeral -t ood:latest
```

Then to deploy (exp):
```
docker run --mount src=/MOUNT_PATH,target=/usr/src/MOUNT_TARGET,type=bind -d ood:latest --framework=tf --model_src_path=MOUNT_TARGE/tf_resnet_50 --model_metadata="{'type': 'tf', 'arch': 'resnet50'}" --data_uri=examples/flower_photos_small.tar.gz --data_metadata="{'img_height': 224, 'img_width': 224, 'batch_size': 32, 'normalize': 255}" --inference_service=in-memory --model_dest_path=MOUNT_DEST/tf_resnet_ood
```

More details on parameters to pass to the ephemeral container can be found [here](ood_enabler/cli/README.md)


## Using a Certainty enabled Model in Openshift Rhods <a name="rhods"></a>
Certainty enabled models can be served in model serving environments such as Red Hat Open Data Science [(RHODS)](https://developers.redhat.com/products/red-hat-openshift-data-science/overview)

After enabling a model for certainty using the model, and saving to a desired location accessible to RHODS, the model can deployed following the regular steps detailed [here.](https://access.redhat.com/documentation/en-us/red_hat_openshift_data_science/1/html/working_on_data_science_projects/model-serving-on-openshift-data-science_model-serving#deploying-a-model-in-openshift-data-science_model-serving)

A sample certainty enabled model to test deployment with RHODS can be found [here](https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/model.onnx). Once deployed successfully, the certainty enabled model can be used for inference using the REST `infer` endpoint provided by RHODS when deployed.

#### Infer Output
A sample input to provide to the deployed model's `infer` endpoint can be found [here](https://public-test-rhods.s3.us-east.cloud-object-storage.appdomain.cloud/input.json). In addition to the model's original ouput in the returned `outputs` json , the deployed certainty enabled model will provide a new output named `ood_scores`, which list the model certainty score for the input to inference. Providing the sample input to the deployed sample should produce output similar to below

exp:
```
{"model_name":"onnx-ood__isvc-b605b06hfd","model_version":"1",
"outputs":[{"name":"logits","datatype":"FP32","shape":[1,1000],"data":[2.857916,2.0202837,...]},
  {"name":"ood_scores","datatype":"FP32","shape":[1,1],"data":[1]}]}
```

## Contributing

Please see the [CONTRIBUTING.md]([./CONTRIBUTING.md](https://github.com/trustyai-explainability/trustyai-explainability/blob/main/CONTRIBUTING.md)) file for more details on how to contribute to this project.

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](./LICENSE) file for details.
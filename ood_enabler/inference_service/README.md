# Inference service

1. [Currently supported architectures and backends](#supported-architectures-and-backends)
2. [Roadmap for extending support to other model architectures and frameworks](#roadmap)
3. [Descriptions and setup instructions](#descriptions-and-setup-instructions)
    1. [In-memory inference](#in-memory-inference)
    2. [KServe inference](#kserve-inference)
    3. [RHODS inference](#rhods-inference)

## Currently supported architectures and backends

The following tables show the supported architectures and inference backends. The reason that some architectures are supported by the native ML backend but not by KServe/RHODS is because there is currently no support for converting these models to ONNX format that is used by KServe/RHODS.

#### PyTorch
| Architecture | Native ML Backend     | KServe on K8s | RHODS |
|---------------|---------| --- |-----------------|
| Resnet18, 50      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Squeezenet     | :heavy_check_mark: ||          |
| VGG16      | :heavy_check_mark: ||          |
| Shufflenet v2      | :heavy_check_mark: ||          |
| Mobilenet v2      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Resnext50      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Wide Resnet50      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Mnasnet     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Vision Transformer (ViT)     | :heavy_check_mark: | | |
| Swin Transformer (swin)     | :heavy_check_mark: | | |
| Yolo v5     | :heavy_check_mark: |  | |


#### TensorFlow
| Architecture | Native ML Backend     | KServe on K8s | RHODS |
|---------------|---------| --- |-----------------|
| Resnet18, 50      | :heavy_check_mark: |:heavy_check_mark:|:heavy_check_mark:|
| VGG16     | :heavy_check_mark: |:heavy_check_mark:|:heavy_check_mark:|
| Efficientnet v2      | :heavy_check_mark: |:heavy_check_mark:|:heavy_check_mark:|
| Mobilenet v2      | :heavy_check_mark: |:heavy_check_mark:|:heavy_check_mark:|
| Mnasnet     | :heavy_check_mark: |:heavy_check_mark:|:heavy_check_mark:|


## Roadmap for extending support to other model architectures and frameworks

The following table shows our current plan for extending support to other popular architectures and frameworks as well as plans for rigorous benchmarking of supported architectures and frameworks.

| Model Source | Model Format Support | Instance vs Distribution | Algorithm | Current Status & Roadmap | Benchmarking |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Torchvision | PyTorch | Instance | Energy | Currently support Resnet, Mobilenet, Resnext50, Mnasnet, Vision Transformer (ViT), Swin-Transformer (swin), Yolo v5 | Ongoing |
| TF Model Types | TensorFlow | Instance | Energy | Currently support Resnet, VGG16, Mobilenet, Mnasnet | Ongoing |
| Huggingface | PyTorch | Instance | Energy | Plan to support Roberta & Distilbert in Q3 | TBD |
| ONNX Model Types | ONNX | Instance | Energy | Plan to support Resnet & other popular CNNs in Q3 | TBD |
|   | Xgboost |   |   |   |   |
|   | Sklearn |   |   |   |   |





## Descriptions and setup instructions

### In-memory inference

The in-memory inference calls native Pytorch or Tensorflow inference backends.

### KServe inference

The KServe inference deploys a KServe inference service with the given model in a Kubernetes environment. Currently only tested with local Kubernetes running with Kind. See `examples/pytorch_example_resnet50_kserve.py` for an example of how to use it.

The following environment variables need to be specified:
```
IBM_API_KEY_ID=
IAM_SERVICE_ID=
ENDPOINT=https://s3.us-east.cloud-object-storage.appdomain.cloud/
IBM_AUTH_ENDPOINT=https://iam.cloud.ibm.com/oidc/token
KSERVE_IP_PORT=127.0.0.1:8080
K8S_CONTEXT=kind-kind
K8S_NAMESPACE=kserve-test
K8S_SA_NAME=sa
```

The bucket name needs to be passed into `OODEnabler.ood_enable()`, as in:
```
OODEnabler.ood_enable(model, data_handler, inference_service=InferenceServiceType.KSERVE, bucket='rhods')
```

S3 credentials and service account for accessing COS needs to be running in Kubernetes, with the name `sa` (or any other name specified in `K8S_SA_NAME`). See https://kserve.github.io/website/0.9/modelserving/storage/s3/s3/#create-s3-secret for explanation. Currently, we only support S3 storage for the inference backend.

The value `K8S_NAMESPACE` can be replaced with the desired Kubernetes namespace where the inference service will be deployed.

If it is desirable not to use environment variables, the COS and KServeInference parameters can also be passed as keyword arguments of `OODEnabler.ood_enable()` directly. When passed to this function directly, the corresponding keywords are:
```
api_key
service_instance_id
endpoint
auth_endpoint
alt_ip_with_port
k8s_context
k8s_namespace
k8s_sa_name
```

The local Kubernetes should operate in port forwarding mode by running these two lines from https://kserve.github.io/website/0.9/get_started/first_isvc/#4-determine-the-ingress-ip-and-ports:
```
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80
```  

### RHODS inference

For inference using Red Hat OpenShift Data Science (RHODS), access to RHODS UI is a prerequisite. Credentials for writing into Cloud Object Storage (COS) are also needed.

To begin, first create data science project and set up data connection and model server:

![rhods_setup](https://media.github.ibm.com/user/18857/files/0dc22804-1a29-4baa-ab52-793d78db5897)


After completing the above steps, it would be best to wait for a few minutes to ensure that the service is up and running.

When running the library locally, logging into the OpenShift cluster via `oc login ...` is needed before the library can deploy models in RHODS for inference.

For using our library, the following environment variables need to be specified:
```
IBM_API_KEY_ID=
IAM_SERVICE_ID=
ENDPOINT=https://s3.us-east.cloud-object-storage.appdomain.cloud/
IBM_AUTH_ENDPOINT=https://iam.cloud.ibm.com/oidc/token
RHODS_PROJECT=ood-test
RHODS_RUNTIME=model-server-ood-test
RHODS_STORAGE_KEY=aws-connection-ood-cos
```
The last three variables are filled according to the configurations with names shown in the above picture. They need to be replaced accordingly if you use different names. For the variable `RHODS_RUNTIME`, it is usually equal to `model-server-$RHODS_PROJECT`.

If it is desirable not to use environment variables, the COS and RHODS parameters can also be passed as keyword arguments of `OODEnabler.ood_enable()` directly. When passed to this function directly, the corresponding keywords are:
```
api_key
service_instance_id
endpoint
auth_endpoint
alt_ip_with_port
rhods_project
rhods_runtime
rhods_storage_key
```

See `examples/pytorch_example_resnet50_rhods.py` for an example on how to use our library with RHODS inference backend. The main step is to add `inference_service=InferenceService.RHODS` and the bucket name as arguments when calling `OODEnabler.ood_enable()`, as in
```
OODEnabler.ood_enable(model, data_handler, inference_service=InferenceServiceType.RHODS, bucket='rhods')
```

The deployed inference services are temporary and have the name `tmp-` followed by a hash code. Currently, these temporary services do not show up in the RHODS UI, but it will be automatically deleted whenever the inference using all the images in the data handler has been completed. Similarly, for each inference service, a temporary folder for model storage will be created in COS, with a parent folder with name `temporary_models` and subfolders with hash code names inside it. The folder will be also deleted when the inference finishes. If the program is interrupted and ends gracefully, the inference service and temporary folder will be also deleted.

Note: Occasionally, call to RHODS may fail due to networking or other issues. Re-running the program usually works.

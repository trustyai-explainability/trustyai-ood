/*
 * © Copyright IBM Corp. 2024, and/or its affiliates. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the “License”);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
from ood_enabler.inference_service.inference_service import InferenceService
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.util.constants import SavedModelFormat
from ood_enabler.inference_service.data_plane import remote_infer
import ood_enabler.settings as s
from kubernetes import client
from kserve import KServeClient
from kserve import constants
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1ONNXRuntimeSpec
import requests
import atexit


class KserveInference(InferenceService):
    def __init__(self, alt_ip_with_port=None, k8s_context=None, k8s_namespace=None, k8s_sa_name=None, **kwargs):
        """
        Initializes an Kserve InferenceService with Model to run inference
        """
        self.alt_ip_with_port = s.KSERVE_IP_PORT if alt_ip_with_port is None else alt_ip_with_port
        self.k8s_context = s.K8S_CONTEXT if k8s_context is None else k8s_context
        self.k8s_namespace = s.K8S_NAMESPACE if k8s_namespace is None else k8s_namespace
        self.k8s_sa_name = s.K8S_SA_NAME if k8s_sa_name is None else k8s_sa_name

        if self.alt_ip_with_port == '':
            self.alt_ip_with_port = None

        self.kwargs = kwargs

    def infer(self, model, data_handler):
        """
        Performs inference on provided model with dataset

        :param model: model to use for inference
        :type model: `model.ModelWrapper`
        :param data_handler: data to be inferred
        :type data_handler: `data.Datahandler`
        :return: stats for normalizing data
        """

        model_store_cos = ModelStore.from_cos(**self.kwargs)

        with model_store_cos.upload_temporary(model, 'temporary_models', saved_model_format=SavedModelFormat.ONNX) as (full_path, model_path, short_hash_str):

            # print('Inference model path:', full_path)

            api_version = constants.KSERVE_GROUP + '/' + 'v1beta1'

            isvc_name = 'tmp-' + short_hash_str
            isvc_instance = V1beta1InferenceService(api_version=api_version,
                                                       kind=constants.KSERVE_KIND,
                                                       metadata=client.V1ObjectMeta(
                                                           name=isvc_name,  # seems model name cannot be too long
                                                           namespace=self.k8s_namespace),
                                                       spec=V1beta1InferenceServiceSpec(
                                                           predictor=V1beta1PredictorSpec(
                                                               service_account_name=self.k8s_sa_name,
                                                               onnx=(V1beta1ONNXRuntimeSpec(
                                                                   storage_uri='s3://' + full_path))))
                                                       )

            k_serve = KServeClient(context=self.k8s_context)

            # TODO: service account creation does not work
            # k_serve.set_credentials(storage_type='S3', namespace=namespace, service_account='sa-tmp', # + short_hash_str,
            #                         s3_endpoint='s3.us-east.cloud-object-storage.appdomain.cloud',
            #                         s3_profile='default',
            #                         s3_use_https='1',
            #                         s3_verify_ssl='0'
            #                         )

            def delete_service():
                k_serve.delete(isvc_name, namespace=self.k8s_namespace)
            atexit.register(delete_service)  # Delete service if program ends unexpectedly

            print('KServe inference service ' + isvc_name + ' being created...')
            k_serve.create(isvc_instance)
            k_serve.wait_isvc_ready(isvc_name, namespace=self.k8s_namespace)
            print('KServe inference service ' + isvc_name + ' ready')

            isvc_info = k_serve.get(isvc_name, namespace=self.k8s_namespace, watch=False, timeout_seconds=120)

            host_name = isvc_info['status']['url'].replace('http://', '').replace('https://', '')
            # print(host_name)

            if self.alt_ip_with_port is None:
                dest_url = isvc_info['status']['url']  # Note: this can possibly be different if running in a pod, need to check
            else:
                dest_url = 'http://' + str(self.alt_ip_with_port)

            r = requests.get(dest_url + '/v2/models/ood', headers={'Host': host_name})
            print(r.content)  # Shows service info, not needed for inference

            inference_results = remote_infer(model, data_handler, dest_url + '/v2/models/ood/infer', headers={'Host': host_name})

            delete_service()
            atexit.unregister(delete_service)

        return inference_results

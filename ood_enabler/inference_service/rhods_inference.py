"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
from ood_enabler.inference_service.inference_service import InferenceService
from ood_enabler.storage.model_store import ModelStore
from ood_enabler.util.constants import SavedModelFormat
from ood_enabler.inference_service.data_plane import remote_infer
import ood_enabler.settings as s
import os
from kubernetes import config
from openshift.dynamic import DynamicClient
import time
import atexit


class RhodsInference(InferenceService):
    def __init__(self, rhods_project=None, rhods_runtime=None, rhods_storage_key=None, **kwargs):
        """
        Initializes an Kserve InferenceService with Model to run inference
        """
        self.rhods_project = s.RHODS_PROJECT if rhods_project is None else rhods_project
        self.rhods_runtime = s.RHODS_RUNTIME if rhods_runtime is None else rhods_runtime
        self.rhods_storage_key = s.RHODS_STORAGE_KEY if rhods_storage_key is None else rhods_storage_key

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

            k8s_client = config.new_client_from_config()
            dyn_client = DynamicClient(k8s_client)

            kserve_service = dyn_client.resources.get(api_version='serving.kserve.io/v1beta1', kind='InferenceService')

            isvc_name = 'tmp-' + short_hash_str
            if model_path.startswith('/'):
                model_path = model_path[1:]
            file_path = (model_path + '/ood/1/model.onnx').replace('//', '/')  # Currently hardcoded to support KServe Triton repository structure used in KServe

            service = {
                'apiVersion': 'serving.kserve.io/v1beta1',
                'kind': 'InferenceService',
                'metadata': {
                    'name': isvc_name,
                    'annotations': {
                        'serving.kserve.io/deploymentMode': 'ModelMesh'
                    }
                },
                'spec': {
                    'predictor': {
                        'model': {
                            'modelFormat': {'name': 'onnx'},
                            'runtime': self.rhods_runtime,
                            'storage': {
                                'key': self.rhods_storage_key,
                                'path': file_path
                            }
                        }
                    }
                }
            }

            def delete_service():
                kserve_service.delete(name=isvc_name, namespace=self.rhods_project)
            atexit.register(delete_service)  # Delete service if program ends unexpectedly

            print('RHODS inference service ' + isvc_name + ' being created...')
            kserve_service.create(body=service, namespace=self.rhods_project)
            ready = False
            while not ready:
                status_list = kserve_service.get(name=isvc_name, namespace=self.rhods_project).status.conditions
                ready = True
                for s in status_list:
                    if s.status != 'True' or not s.status:
                        ready = False
                if not ready:
                    time.sleep(1.0)

            print('RHODS inference service ' + isvc_name + ' ready')
            time.sleep(1.0)  # There seems a bit delay before the route gets fully set up, so wait for 1 sec here

            v1_routes = dyn_client.resources.get(api_version='route.openshift.io/v1', kind='Route')
            routes = v1_routes.get(name=isvc_name, namespace=self.rhods_project)

            inference_results = remote_infer(model, data_handler, 'https://' + routes.spec.host + routes.spec.path + '/infer')

            delete_service()
            atexit.unregister(delete_service)

        return inference_results

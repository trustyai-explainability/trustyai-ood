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


class InMemoryInference(InferenceService):
    def __init__(self, **kwargs):
        pass

    def infer(self, model, data_handler):
        """
        Performs inference on provided model with dataset

        :param model: model wrapper to use for inference
        :type model: `model.ModelWrapper`
        :param data_handler: data to be inferred
        :type data_handler: `data.Datahandler`
        :return: stats for normalizing data
        """
        return model.infer(data_handler)

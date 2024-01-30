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
import numpy as np
from ood_enabler.model_modifier.model_modifier import ModelModifier
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import Model
from ood_enabler.settings import DEFAULT_OOD_THRESH_PERCENTILE


class TFModifier(ModelModifier):
    """
    Class to transform Tensorflow/Keras model for OOD enablement
    """

    def add_ood_layer(self, model_wrapper):
        """
        Based on input method, add OOD layer to the model

        :param model_wrapper: model to embed with OOD layer
        :type model_wrapper: `model_wrapper.Model`
        return: transformed OOD Model
        """
        model = model_wrapper.model
        model.get_layer("predictions").activation = None

        ood_model = Model(inputs=model.input,
                          outputs=[model.get_layer("predictions").output,
                                   layers.Lambda(lambda t: tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(t), -1)), axis=1),  # Add one more dimension to support ONNX in KServe
                                                 name='ood_scores')(model.get_layer("predictions").output)])
        model_wrapper.model = ood_model

        return model_wrapper

    def add_normalization_layer(self, model_wrapper, inference_results):
        """
        Based on inference results (forward pass), add normalization layer for OOD

        :param model_wrapper: model to embed with normalization layer
        :type model_wrapper: `model_wrapper.Model`
        :param inference_results: results from transformed OOD model; should include ood_score

        """
        model = model_wrapper.model
        model.layers[-1]._name = 'ood_scores_intermediate'  # To make sure that the final OOD layer has the same name, change this to a different name
                                                            # This edits the internal variable, which is not ideal, but couldn't find a different approach for now

        # threshold = min(e for b in inference_results for e in b[1])[0]

        flt_enrg_scr_lst = [e for b in inference_results for e in b[1]]
        if 'ood_thresh_percentile' not in model_wrapper.model_metadata:
            threshold = np.percentile(flt_enrg_scr_lst, DEFAULT_OOD_THRESH_PERCENTILE)
        else:
            threshold = np.percentile(flt_enrg_scr_lst, model_wrapper.model_metadata['ood_thresh_percentile'])

        norm_model = Model(inputs=model.input, outputs=[model.get_layer("predictions").output,
                                                layers.Lambda(lambda t: tf.divide(tf.clip_by_value(t, 0, threshold), threshold),
                                                              name='ood_scores')(
                                                    model.get_layer("ood_scores_intermediate").output)])
        model_wrapper.model = norm_model
        return model_wrapper

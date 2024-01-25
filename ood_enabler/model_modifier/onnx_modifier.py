"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
import numpy as np
from ood_enabler.model_modifier.model_modifier import ModelModifier
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import Model
import onnx


class ONNXModifier(ModelModifier):
    """
    Class to transform ONNX model for OOD enablement
    """

    def add_ood_layer(self, model_wrapper):
        """
        Based on input method, add OOD layer to the model

        :param model_wrapper: model to embed with OOD layer
        :type model_wrapper: `model_wrapper.Model`
        return: transformed OOD Model
        """
        model = model_wrapper.model
        last_layer = model.graph.node[-1]
        log_sum_exp_node = onnx.helper.make_node('ReduceLogSumExp', inputs=[last_layer.output[0]], outputs=['log_sum_exp_output'], keepdims=1)
        original_output_node = onnx.helper.make_node('Identity', inputs=[last_layer.output[0]], outputs=['original_output'])

        model.graph.node.extend([log_sum_exp_node, original_output_node])
        model.graph.output[0].name = 'original_output'
        model.graph.output.extend([onnx.helper.make_tensor_value_info('log_sum_exp_output', onnx.TensorProto.FLOAT, None)])
        model_wrapper.model = model

        return model_wrapper


    def add_normalization_layer(self, model_wrapper, inference_results):
        """
        Based on inference results (forward pass), add normalization layer for OOD

        :param model_wrapper: model to embed with normalization layer
        :type model_wrapper: `model_wrapper.Model`
        :param inference_results: results from transformed OOD model; should include ood_score

        """

        return model_wrapper

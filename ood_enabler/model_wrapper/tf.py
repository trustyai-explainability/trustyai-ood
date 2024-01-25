"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
import tensorflow as tf
import numpy
import os
import datetime
from tempfile import TemporaryDirectory
from ood_enabler.model_wrapper.model import ModelWrapper
from ood_enabler.util.constants import MLBackendType, SupportedModelArch
from ood_enabler.util.constants import SavedModelFormat
from ood_enabler.data.data_handler import DataHandler
from ood_enabler.util.archiver import extract_archive
import tf2onnx
from tqdm import tqdm


class TFWrapper(ModelWrapper):
    def __init__(self, model_metadata, model=None, path=None, **kwargs):
        """
        Initializes wrapper for model

        :param model_metadata: metadata about model (i.e. architecture, type)
        :type model_metadata: `dict`
        :param model: loaded model (only one of model and path needs to be specified)
        :param path: path to be loaded (only one of model and path needs to be specified)
        """
        super().__init__(model_metadata, model, path, **kwargs)
        self.model_metadata['type'] = MLBackendType.TF

    def load(self, path, bypass_model_check=False, **kwargs):
        """
        loads in memory a saved model TF model from path

        :param path: path to where model is accessible
        :param bypass_model_check: if True, checking of whether the architecture is supported is disabled; can be useful for loading an OOD-enabled model for testing purpose
        :return: the loaded model
        """
        with TemporaryDirectory() as tmpdir:
            # catch case of model saved in format for kserve
            if os.path.isdir(path):
                if os.path.exists(os.path.join(path, "0001")):
                    path = os.path.join(path, "0001")

            # assume zip or .tar.gz
            else:
                path = extract_archive(path, tmpdir)

            return tf.keras.models.load_model(path, **kwargs)

    def save(self, path, saved_model_format=SavedModelFormat.NATIVE, **kwargs):
        """
        saves underlying TF model to provided path

        :param path: path to where model should be saved
        :param saved_model_format: file/directory format of saved model
        :type saved_model_format: `util.constants.SavedModelFormat`
        :return: path to where model is saved
        """

        if saved_model_format == SavedModelFormat.NATIVE:
            timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))
            outer_path = os.path.join(path, "ood_tf_{}".format(timestamp))
            inner_path = os.path.join(path, outer_path, "0001")
            self.model.save(inner_path, **kwargs)

            return outer_path

        elif saved_model_format == SavedModelFormat.ONNX:
            path_config = os.path.join(path, 'ood')
            path_model = os.path.join(path_config, '1')
            os.makedirs(path_config)
            os.makedirs(path_model)

            path_onnx = os.path.join(path_model, "model.onnx")

            config_str = \
                '''
                name: "ood"
                backend: "onnxruntime"
                max_batch_size : 256
                input [
                  {
                    name: "input"
                    data_type: TYPE_FP32
                    dims: INPUT_DIMS
                  }
                ]
                output [
                  {
                    name: "predictions"
                    data_type: TYPE_FP32
                    dims: [  NUM_CLASSES]
                  }
                ]
                '''

            input_dims = self.model.layers[0].get_config()['batch_input_shape']
            config_str = config_str.replace('INPUT_DIMS', str(list(input_dims)[1:]))
            config_str = config_str.replace('NUM_CLASSES', str(self.model.get_layer('predictions').output_shape[1]))

            dummy_input = tf.zeros((10, *input_dims[1:]))
            output = self.model(dummy_input)
            if isinstance(output, tuple) or isinstance(output, list):
                config_str += \
                '''
                output [
                  {
                    name: "ood_scores"
                    data_type: TYPE_FP32
                    dims: [ 1]
                  }
                ]
                '''

            tf2onnx.convert.from_keras(self.model, input_signature=[tf.TensorSpec(input_dims, name='input')], output_path=path_onnx)

            with open(os.path.join(path_config, 'config.pbtxt'), 'w') as f:
                f.write(config_str)

            return path

        else:
            raise ValueError('Unsupported model format: ' + saved_model_format)

    def infer(self, data_handler: DataHandler, func=None):
        """
        infers on underlying TF model

        :param data_handler: datahandler with to be inferred
        :type data_handler: `data.data_handler`
        :param func: function callback for inference; if None, use native inference call from ML backend
        :return: inference results
        """
        dataset = data_handler.get_dataset()
        if isinstance(dataset, numpy.ndarray):
            return self.model.predict(dataset) if func is None else func(dataset)

        else:
            all_outputs = []
            for data in tqdm(dataset, desc='Inference progress'):
                # print(data.shape)
                outputs = self.model.predict(data.numpy()) if func is None else func(data.numpy())
                all_outputs.append(outputs)

            return all_outputs

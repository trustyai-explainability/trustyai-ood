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
import datetime
import os
import torch
import torchvision
import transformers
from ood_enabler.data.data_handler import DataHandler
from ood_enabler.model_wrapper.model import ModelWrapper
from ood_enabler.util.constants import MLBackendType, SupportedModelArch
from ood_enabler.util.base_model import get_pytorch_base_model
from ood_enabler.exceptions.exceptions import UnknownMLArch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from model_archiver.arg_parser import ArgParser
from model_archiver.model_packaging_utils import ModelExportUtils
from model_archiver.model_packaging import package_model

import zipfile
from tempfile import TemporaryDirectory

from ood_enabler.util.constants import SavedModelFormat
from onnx2pytorch import ConvertModel
import onnx
from tqdm import tqdm


class PytorchWrapper(ModelWrapper):
    def __init__(self, model_metadata, model=None, path=None, **kwargs):
        """
        Initializes wrapper for model

        :param model_metadata: metadata about model (i.e. architecture, type)
        :type model_metadata: `dict`
        :param model: loaded model (only one of model and path needs to be specified)
        :param path: path to be loaded (only one of model and path needs to be specified)
        """
        super().__init__(model_metadata, model, path, **kwargs)
        self.model_metadata['type'] = MLBackendType.PYTORCH

    @staticmethod
    def base_arch_layers(model_metadata):

        base_model = ''
        
        if model_metadata['arch'] == 'resnet18':
            base_model = torchvision.models.resnet18(pretrained=True)
        elif model_metadata['arch'] == 'resnet50':
            base_model = torchvision.models.resnet50(pretrained=True)
        elif model_metadata['arch'] == 'squeezenet1_0':
            base_model = torchvision.models.squeezenet1_0(pretrained=True)
        elif model_metadata['arch'] == 'vgg16':
            base_model = torchvision.models.vgg16(pretrained=True)
        elif model_metadata['arch'] == 'densenet161':
            base_model = torchvision.models.densenet161(pretrained=True)
        elif model_metadata['arch'] == 'shufflenet_v2_x1_0':
            base_model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        elif model_metadata['arch'] == 'mobilenet_v2':
            base_model = torchvision.models.mobilenet_v2(pretrained=True)
        elif model_metadata['arch'] == 'resnext50_32x4d':
            base_model = torchvision.models.resnext50_32x4d(pretrained=True)
        elif model_metadata['arch'] == 'wide_resnet50_2':
            base_model = torchvision.models.wide_resnet50_2(pretrained=True)
        elif model_metadata['arch'] == 'mnasnet1_0':
            base_model = torchvision.models.mnasnet1_0(pretrained=True)
        elif model_metadata['arch'] == 'vit_b_16':
            base_model = torchvision.models.vit_b_16(pretrained=True)
        elif model_metadata['arch'] == 'vit_b_32':
            base_model = torchvision.models.vit_b_32(pretrained=True)
        elif model_metadata['arch'] == 'vit_l_16':
            base_model = torchvision.models.vit_l_16(pretrained=True)
        elif model_metadata['arch'] == 'vit_l_32':
            base_model = torchvision.models.vit_l_32(pretrained=True)
        elif model_metadata['arch'] == 'vit_h_14':
            base_model = torchvision.models.vit_h_14(pretrained=True)
        elif model_metadata['arch'] == 'swin_t':
            base_model = torchvision.models.swin_t(pretrained=True)
        elif model_metadata['arch'] == 'swin_s':
            base_model = torchvision.models.swin_s(pretrained=True)
        elif model_metadata['arch'] == 'swin_b':
            base_model = torchvision.models.swin_b(pretrained=True)
        elif model_metadata['arch'] == 'swin_v2_t':
            base_model = torchvision.models.swin_v2_t(pretrained=True)
        elif model_metadata['arch'] == 'swin_v2_s':
            base_model = torchvision.models.swin_v2_s(pretrained=True)
        elif model_metadata['arch'] == 'swin_v2_b':
            base_model = torchvision.models.swin_v2_b(pretrained=True)    
        elif model_metadata['arch'] == 'roberta':
            MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
            base_model = AutoModelForSequenceClassification.from_pretrained(MODEL, torchscript=True)

        if not base_model == '':
            base_model_layers = [(i[0]) for i in base_model.named_modules() if not list(i[1].named_children())]
            print('returning base model ', base_model)
            return base_model_layers
        else:
            print('returning empty layer list\n')
            return ['']

    @staticmethod
    def layer_name_similarity(base_model_layers, scripted_model_layers):
        base_len = len(base_model_layers)
        scrp_len = len(scripted_model_layers)
        for i in range(0, min(base_len, scrp_len)):
            if base_model_layers[i] not in scripted_model_layers[i]:
                print("{} mismatch with {}\n".format(base_model_layers[i], scripted_model_layers[i]))
                return False
        return True

    @staticmethod
    def model_arch_diff(base_model_layers, scripted_model_layers):
        if len(scripted_model_layers) != len(base_model_layers):
            print("Not similar base model architecture as different number of layers\n")
        elif scripted_model_layers[-1] != base_model_layers[-1]:
            print("Not similar base model architecture as unexpected last layer\n")
        elif not PytorchWrapper.layer_name_similarity(base_model_layers, scripted_model_layers):
            print("Not similar base model architecture as different types of layers\n")
        else:
            print("Compatible base model architecture\n")
            return False
        return True

    def load_checkpoint(self, checkpoint):
        """
        loads in a checkpoint to the architecture specified in the model metadata

        :param checkpoint: the model checkpoint
        :type checkpoint: `dict`
        :return: the loaded model
        """
        base_model = get_pytorch_base_model(self.model_metadata['arch'])
        base_model.load_state_dict(checkpoint)
        return base_model

    def load(self, path, bypass_model_check=False, **kwargs):
        """
        loads in memory a saved model Pytorch from path

        :param path: path to where model is accessible
        :param bypass_model_check: if True, checking of whether the architecture is supported is disabled; can be useful for loading an OOD-enabled model for testing purpose
        :return: the loaded model
        """

        if os.path.isdir(path):
            paths_all = []
            for (dir_path, dir_names, file_names) in os.walk(path):
                for f in file_names:
                    paths_all.append(os.path.join(dir_path, f))
        else:
            paths_all = [path]

        model = None
        for p in paths_all:
            try:
                try:
                    model = torch.load(p, **kwargs)
                    if isinstance(model, dict):
                        try:
                            model = self.load_checkpoint(model)
                        except:
                            model = None
                            raise ValueError()
                except:
                    try:
                        onnx_model = onnx.load(p)
                        model = ConvertModel(onnx_model)
                    except:
                        raise ValueError()
            except:
                with TemporaryDirectory() as tmpdir:
                    try:
                        with zipfile.ZipFile(p, 'r') as zip_ref:
                            zip_ref.extractall(tmpdir)

                        for (dir_path, dir_names, file_names) in os.walk(tmpdir):
                            for f in file_names:
                                path_new = os.path.join(dir_path, f)
                                try:
                                    model = torch.load(path_new, **kwargs)
                                    if isinstance(model, dict):
                                        # model checkpoint state_dict
                                        model = self.load_checkpoint(model)
                                    break
                                except:
                                    try:
                                        onnx_model = onnx.load(path_new)
                                        model = ConvertModel(onnx_model)
                                        break
                                    except:
                                        pass

                            if model is not None:
                                break
                    except zipfile.BadZipfile:
                        pass
            if model is not None:
                break

        if model is None:
            raise Exception('No supported model file found')

        if not bypass_model_check:
            scripted_model_layers = [(i[0]) for i in model.named_modules() if not list(i[1].named_children())]
            print("Number of layers for model from path:{}".format(len(scripted_model_layers)))
            base_model_layers = PytorchWrapper.base_arch_layers(self.model_metadata)
            print("Number of layers for base reference model:{}".format(len(base_model_layers)))

            if PytorchWrapper.model_arch_diff(base_model_layers, scripted_model_layers):
                raise UnknownMLArch("Given model architecture is not supported")

        return model

    def save(self, path, saved_model_format=SavedModelFormat.TORCHSCRIPT, **kwargs):
        """
        saves underlying Pytorch model to provided path

        :param path: path to where model should be saved
        :param saved_model_format: file/directory format of saved model
        :type saved_model_format: `util.constants.SavedModelFormat`
        :return: path to where model is saved
        """

        # TODO: Make model name a configurable parameter

        self.model.eval()  # Only save evaluation models, in all cases

        timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))

        if saved_model_format == SavedModelFormat.NATIVE or saved_model_format == SavedModelFormat.TORCHSCRIPT:
            if self.model_metadata['arch'] == "roberta":
               
                dummy_input1 = torch.zeros(size=(32, 512), dtype=torch.long)
                dummy_input2 = torch.zeros(size=(32, 512), dtype=torch.long)
                print("Saving hf model out with torchscript")
                path_pt = os.path.join(path, "ood_pytorch_{}.pt".format(timestamp))
                model_full = torch.jit.trace(self.model,[dummy_input1, dummy_input2])
                model_full.save(path_pt)
                return path_pt
            else: 
                path_pt = os.path.join(path, "ood_pytorch_{}.pt".format(timestamp))
                model_full = torch.jit.script(self.model)
                model_full.save(path_pt)
                return path_pt
        elif saved_model_format == SavedModelFormat.STATE_DICT:
            path_pt = os.path.join(path, "ood_pytorch_{}.pt".format(timestamp))
            torch.save(self.model.state_dict(), path_pt)
            return path_pt

        elif saved_model_format == SavedModelFormat.ONNX:
            if self.model_metadata['arch'] == "roberta":
                path_config = os.path.join(path, 'ood')
                path_model = os.path.join(path_config, '1')
                os.makedirs(path_config)
                os.makedirs(path_model)
                path_onnx = os.path.join(path_model, "model.onnx")
                dummy_input = torch.zeros(size=(10, 512), dtype=torch.long)
                dummy_input1 = torch.zeros(size=(32, 512), dtype=torch.long)
                dummy_input2 = torch.zeros(size=(32, 512), dtype=torch.long)

                config_str = \
                    '''
                    name: "ood"
                    backend: "onnxruntime"
                    max_batch_size : 256
                    input [
                    {
                        name: "input"
                        data_type: TYPE_FP32
                        dims: [ 3, -1, -1 ]
                    }
                    ]
                    output [
                    {
                        name: "logits"
                        data_type: TYPE_FP32
                        dims: [  NUM_CLASSES]
                    }
                    ]
                    '''
                output = self.model(dummy_input)
                # output = self.model((dummy_input1, dummy_input2))
                if isinstance(output, tuple) or isinstance(output, list):
                    num_classes = output[0].shape[-1]
                    output_names = ['logits', 'ood_scores']
                    dynamic_axes = {'input': {0: 'batch_size', 1: 'sentence_length'},
                                                'logits': {0: 'batch_size'}, 'ood_scores': {0: 'batch_size'}}

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
                else:
                    num_classes = output.shape[-1]
                    output_names = ['logits']
                    dynamic_axes = {'input': {0: 'batch_size', 1: 'sentence_length'},
                                                'logits': {0: 'batch_size'}}

                model_full = torch.jit.trace(self.model,[dummy_input1, dummy_input2])
                torch.onnx.export(model_full, (dummy_input1, dummy_input2), path_onnx, input_names=['input'], output_names=output_names,
                                dynamic_axes=dynamic_axes)

                with open(os.path.join(path_config, 'config.pbtxt'), 'w') as f:
                    f.write(config_str)

                return path

            else: 
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
                        dims: [ 3, -1, -1 ]
                    }
                    ]
                    output [
                    {
                        name: "logits"
                        data_type: TYPE_FP32
                        dims: [  NUM_CLASSES]
                    }
                    ]
                    '''

                dummy_input = torch.zeros(10, 3, 224, 224)
                output = self.model(dummy_input)
                if isinstance(output, tuple) or isinstance(output, list):
                    num_classes = output[0].shape[-1]
                    output_names = ['logits', 'ood_scores']
                    dynamic_axes = {'input': {0: 'batch_size', 2: 'image_width', 3: 'image_height'},
                                                'logits': {0: 'batch_size'}, 'ood_scores': {0: 'batch_size'}}

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
                else:
                    num_classes = output.shape[-1]
                    output_names = ['logits']
                    dynamic_axes = {'input': {0: 'batch_size', 2: 'image_width', 3: 'image_height'},
                                                'logits': {0: 'batch_size'}}

                config_str = config_str.replace('NUM_CLASSES', str(num_classes))
                model_full = torch.jit.script(self.model)
                torch.onnx.export(model_full, dummy_input, path_onnx, input_names=['input'], output_names=output_names,
                                dynamic_axes=dynamic_axes)

                with open(os.path.join(path_config, 'config.pbtxt'), 'w') as f:
                    f.write(config_str)

                return path
            
        elif saved_model_format == SavedModelFormat.TORCHSERVE:
            with TemporaryDirectory() as tmpdir:
                path_pt = os.path.join(tmpdir, "ood_pytorch_{}.pt".format(timestamp))
                model_full = torch.jit.script(self.model)
                model_full.save(path_pt)

                handler = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       os.pardir, 'static_files', 'pytorch_archiver_handler.py'))

                path_config = os.path.join(path, 'config')
                path_model = os.path.join(path, 'model-store')
                os.makedirs(path_config)
                os.makedirs(path_model)

                model_name = 'ood'  # TODO: make it configurable
                mar_name = "ood_pytorch_{}".format(timestamp)

                config_str = \
                    '''
                    inference_address=http://0.0.0.0:8085
                    management_address=http://0.0.0.0:8085
                    metrics_address=http://0.0.0.0:8082
                    grpc_inference_port=7070
                    grpc_management_port=7071
                    enable_metrics_api=true
                    metrics_format=prometheus
                    number_of_netty_threads=4
                    job_queue_size=10
                    enable_envvars_config=true
                    install_py_dep_per_model=true
                    model_store=/mnt/models/model-store
                    '''
                config_str += 'model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"' + \
                              model_name + \
                              '":{"1.0":{"defaultVersion":true,"marName":"' + \
                              mar_name + '.mar' \
                              '","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":10,"responseTimeout":120}}}}'

                with open(os.path.join(path_config, 'config.properties'), 'w') as f:
                    f.write(config_str)

                args_list = ['--model-name', mar_name,
                             '--serialized-file', path_pt,
                             '--handler', handler,
                             '--export-path', path_model,
                             '-v', '0.1']  # TODO make hardcoded entries configurable

                args = ArgParser.export_model_args_parser().parse_args(args=args_list)
                manifest = ModelExportUtils.generate_manifest_json(args)
                package_model(args, manifest=manifest)

            return path

        else:
            raise ValueError('Unsupported model format: ' + saved_model_format.value)

    def infer(self, data_handler: DataHandler, func=None):
        """
        infers on underlying pytorch model

        :param data_handler: datahandler with dataset to be inferred
        :type data_handler: `data.data_handler`
        :param func: function callback for inference; if None, use native inference call from ML backend
        :return: inference results
        """
        self.model.eval()
        all_outputs = []

        dataset = data_handler.get_dataset()
        

        if isinstance(dataset, torch.utils.data.DataLoader):
            with torch.no_grad():
                for data in tqdm(dataset, desc='Inference progress'):
                    if isinstance(data, list):
                        outputs = self.model(data[0]) if func is None else func(data[0])
                        all_outputs.append(outputs)
                    elif isinstance(data, dict):
                        # case for HF Roberta model 
                        outputs = self.model(data["input_ids"], data["attention_mask"]) 
                        all_outputs.append(outputs)
        else:
            with torch.no_grad():
                all_outputs = self.model(dataset) if func is None else func(dataset)

        return all_outputs

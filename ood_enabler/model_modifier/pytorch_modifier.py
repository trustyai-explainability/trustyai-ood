import torch
from typing import Tuple
from ood_enabler.model_modifier.model_modifier import ModelModifier
from ood_enabler.settings import DEFAULT_OOD_THRESH_PERCENTILE
import torchextractor as tx
from ultralytics import YOLO
import numpy as np

class HFSequential(torch.nn.Sequential):
    """
    Class for wrapping sequential layer for HF Roberta model, so that multiple inputs (tokens and attention masks)
    can be passed to the model. 
    """
    def forward(self, *inputs):
        with torch.no_grad():
            for module in self._modules.values():
                if type(inputs) == tuple and str(module) != "NormalizedOODModelPytorch()" :
                    inputs = module(*inputs)
                else:
                    # normalization layer does not need to have inputs unpacked
                    inputs = module(inputs)
            return inputs

class PytorchModifier(ModelModifier):
    """
    Class for transforming pytorch models for OOD enablement

    """
    class OODModelPytorch(torch.nn.Module):

        def __init__(self):
            super().__init__()

        @staticmethod
        def score(logits, t):
            """
            :param logits: logits of input
            :param t: temperature value
            """
            return t * torch.logsumexp(logits / t, dim=1)

        def forward(self, logit):
            with torch.no_grad():
                temp = torch.tensor(1.0)
                energy_scr_val = self.score(logit, temp)
            return logit, torch.unsqueeze(energy_scr_val, dim=1)   # Add one dimension for ONNX support
   
    class OODModelPytorch_HF_Roberta(torch.nn.Module):
        """
        Class for transforming HF transformer model for OOD enablement
        """

        def __init__(self):
            super().__init__()

        @staticmethod
        def score(logits, t):
            """
            :param logits: logits of input
            :param t: temperature value
            """
            return t * torch.logsumexp(logits / t, dim=1)

        def forward(self, logit):
            logit = logit.logits
            with torch.no_grad():
                temp = torch.tensor(1.0)
                energy_scr_val = self.score(logit, temp)
            return logit, torch.unsqueeze(energy_scr_val, dim=1)   # Add one dimension for ONNX support

    """
    Class for transforming pytorch YOLOv5 models for OOD enablement

    """
    class OODModelPytorch_YOLOv5(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def eng_func(self, a):
            """Average first and last element of a 1-D array"""
            eng_val = np.log(np.sum(np.exp(a)))
            return eng_val

        def layer_based_energy_scr_generator(self, features, num_classes=80):
            batch_sz = (list(features.items())[0][1]).shape[0]
            batch_ood_scr_list = []
            for i in range(batch_sz):
                layer_based_max_eng_scr = []
                for l in features.keys():
                    layer = features[l][i].detach().numpy()
                    anch_1_logit = layer[5:(1*num_classes+5), :, :]
                    anch_2_logit = layer[(1*num_classes+10):(2*num_classes+10), :, :]
                    anch_3_logit = layer[(2*num_classes+15):(3*num_classes+15), :, :]
                    logit_stacked = np.stack((anch_1_logit, anch_2_logit, anch_3_logit), axis=0)
                    max_j_cell = np.max(logit_stacked, axis=0)
                    layer_ood_scr_matrix = np.apply_along_axis(self.eng_func, 0, max_j_cell)
                    layer_ood_scr = (np.max(layer_ood_scr_matrix))
                    layer_based_max_eng_scr.append(layer_ood_scr)
                enr_scr = np.max(layer_based_max_eng_scr)
                batch_ood_scr_list.append(enr_scr)

            return np.array(batch_ood_scr_list)

        def forward(self, model_output_features):
            model_output = model_output_features[0]
            features = model_output_features[1]
            with torch.no_grad():
                energy_scr_val = self.layer_based_energy_scr_generator(features)
                energy_scr_val_ts = torch.from_numpy(energy_scr_val)
            return model_output, torch.unsqueeze(energy_scr_val_ts, dim=1)

    class NormalizedOODModelPytorch(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.ood_threshold = None

        def set_threshold(self, threshold):
            self.ood_threshold = torch.nn.Parameter(threshold, requires_grad=False)
            print("set_threshold for given data handler:{}".format(self.ood_threshold))

        def forward(self, logit_and_ood_score: Tuple[torch.Tensor, torch.Tensor]):
            logit = logit_and_ood_score[0]
            ood_score = logit_and_ood_score[1]
            c = torch.div(ood_score, self.ood_threshold)
            norm_energy_scr_val = torch.clamp(c, min=0.0, max=1.0)
            return logit, norm_energy_scr_val

    def add_ood_layer(self, model_wrapper):
        """
        Based on input method, add OOD layer to the model

        :param model_wrapper: model to embed with OOD layer
        :type model_wrapper: `model_wrapper.Model`
        return: transformed OOD Model
        """
        if model_wrapper.model_metadata['arch'] == 'yolov5l':
            print("In YOLOv5l model condition\n")
            model_org = model_wrapper.model
            layer_lst = tx.list_module_names(model_org)
            print("New feature extractor model is created with layers: {}".format(layer_lst[-3:]))
            model_extractor = tx.Extractor(model_org, layer_lst[-3:])
            model_wrapper.model = model_extractor.eval()
            print("Creating an OOD enabled model for YOLO V5\n")
            ood_model = torch.nn.Sequential(
                model_wrapper.model,
                PytorchModifier.OODModelPytorch_YOLOv5()
            )
        elif model_wrapper.model_metadata['arch'] == 'roberta':
            print("In HF Roberta model condition\n")
            print("Creating an OOD enabled model for HF Roberta\n")
            ood_model = HFSequential(
            model_wrapper.model,
            PytorchModifier.OODModelPytorch_HF_Roberta()
            )
        else:
            print("adding OOD layer pytorch")
            ood_model = torch.nn.Sequential(
            model_wrapper.model,
            PytorchModifier.OODModelPytorch()
            )
        model_wrapper.model = ood_model
        return model_wrapper

    def add_normalization_layer(self, ood_model, inference_results):
        """
        Based on inference results (forward pass), add normalization layer for OOD

        :param model: model to embed with normalization layer
        :type model: `model_wrapper.Model`
        :param inference_results: results from transformed OOD model; should include ood_score

        """
        ergy_score_lst = [i[1] for i in inference_results]
        flt_enrg_scr_lst = [item for sublist in ergy_score_lst for item in sublist]
        if 'ood_thresh_percentile' not in ood_model.model_metadata:
            ood_threshold = np.percentile(flt_enrg_scr_lst, DEFAULT_OOD_THRESH_PERCENTILE)
        else:
            ood_threshold = np.percentile(flt_enrg_scr_lst, ood_model.model_metadata['ood_thresh_percentile'])

        if not isinstance(ood_threshold, torch.Tensor):
            ood_threshold = torch.tensor(ood_threshold)

        norm_layer = PytorchModifier.NormalizedOODModelPytorch()
        norm_layer.set_threshold(ood_threshold)
        if ood_model.model_metadata['arch'] == 'roberta':
            print("In HF Roberta Condtion for normalization layer\n")
            norm_ood_model = HFSequential(
                ood_model.model,
                norm_layer
            )
            ood_model.model = norm_ood_model
        else:
            norm_ood_model = torch.nn.Sequential(
                ood_model.model,
                norm_layer
            )
            ood_model.model = norm_ood_model

        return ood_model


"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler  # Note: Package ts is called by TorchServe service, it is not needed as a requirement for this project
import itertools


class OODImageClassifier(VisionHandler):
    """
    ImageClassifier handler class. This handler takes an image
    and returns the name of object in that image.
    """

    topk = 5
    # These are the standard Imagenet dimensions
    # and statistics
    image_processing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def postprocess(self, data):

        def map_class_to_dict(probs, mapping=None, lbl_classes=None, energy_scores=None):
            """
            Given a list of classes & probabilities, return a dictionary of
            { friendly class name -> probability }
            """
            if not isinstance(probs, list) or not isinstance(energy_scores, list):
                raise Exception("Convert classes to list before doing mapping")

            if mapping is not None and not isinstance(mapping, dict):
                raise Exception("Mapping must be a dict")

            if lbl_classes is None:
                lbl_classes = itertools.repeat(range(len(probs[0])), len(probs))

            results = [
                {
                    (mapping[str(lbl_class)] if mapping is not None else str(lbl_class)): prob
                    for lbl_class, prob in zip(*row)
                }
                for row in zip(lbl_classes, probs)
            ]

            if len(results) != len(energy_scores):
                raise Exception("Energy scores and predictions should have the same dimension")

            new_results = []
            for i in range(len(energy_scores)):
                new_results.append({
                    'predictions': results[i],
                    'OOD_score': energy_scores[i]
                })

            return new_results

        logits, energy_scr = data

        print('**** energy:', energy_scr)
        ps = F.softmax(logits, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        energy_scr = energy_scr.tolist()
        print('**** probs:', probs)
        print('**** classes:', classes)
        return map_class_to_dict(probs, self.mapping, classes, energy_scr)

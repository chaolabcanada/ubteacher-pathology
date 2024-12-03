"""
GradCAM (gradient-weighted class activation mapping) 

@Version: 0.0.1
@Author: Jesse Chao, PhD
@Contact: jesse.chao@sri.utoronto.ca
"""

from typing import Dict, Tuple, List, Set, Iterator

import torch
from torch import nn
import numpy as np
import cv2

from . import train_utils as train_utils


def get_conv_layers(model: object) -> List:
    backbone_conv_layers = []
    for layer_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and "backbone" in layer_name:
            backbone_conv_layers.append(layer_name)
    return backbone_conv_layers


class GradcamMultilayer:
    def __init__(self, model:object, input_image: np.ndarray, target_layers: List):
        self.model = model
        self.input_image = input_image
        self.gradients = []
        self.activations = []
        self.handles = []
        self.target_layers = target_layers
        self._register_hook()
    
    def _save_activation(self, module, input, output):
        #print(f"activations: {output[0].shape}")
        self.activations.append(output.detach().cpu())

    def _save_gradient(self, module, input, output):
        # Gradients are computed in reverse
        def _store_grad(grad):
            #print(f"gradients: {grad[0].shape}")
            self.gradients = [grad.detach().cpu()] + self.gradients
        self.handles.append(
            output.register_hook(_store_grad)
        )

    def _register_hook(self):
        for (layer_name, module) in self.model.named_modules():
            for i in self.target_layers:
                if layer_name == i:
                    self.handles.append(
                        module.register_forward_hook(self._save_activation)
                    )
                    self.handles.append(
                        module.register_forward_hook(self._save_gradient)
                    )
    
    def release(self):
        for handle in self.handles:
            handle.remove()

    def compute_cam_per_layer(self):
        activations_list = [a[0].data.numpy() for a in self.activations]
        grads_list = [g[0].data.numpy() for g in self.gradients]
        cam_per_target_layer = []
        
        for i in range(len(self.target_layers)):
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
            #print(f"layer_grads: {layer_grads.shape}")
            layer_weights = np.mean(layer_grads, axis=(1, 2))
            #print(f"layer_weights: {layer_weights.shape}")
            #print(f"layer activations: {layer_activations.shape}")
            weighted_activations = layer_activations * layer_weights[:, np.newaxis, np.newaxis]
            layer_cam = np.sum(weighted_activations, axis=0)  # [H,W]
            #print(f"layer_cam b4 postprocessing: {layer_cam.shape}")
            # Postprocess
            layer_cam = np.maximum(layer_cam, 0)  # ReLU
            layer_cam -= np.min(layer_cam)
            layer_cam /= np.max(layer_cam)
            # Resize to input image
            resized_layer_cam = cv2.resize(layer_cam, (self.input_image.shape[1], self.input_image.shape[0]))
            # Append to list
            cam_per_target_layer.append(resized_layer_cam)
        return cam_per_target_layer
    
    def get_cam(self):
        cams = self.compute_cam_per_layer()
        # Stack cams from all requested layers together
        cams = np.stack(cams, axis=0)
        #print(f"after concatenating: {cams.shape}")
        # Combine all cams and normalize
        combined_cam = np.maximum(cams, 0)
        combined_cam = np.mean(cams, axis=0)
        #print(f"final cam: {final_cam.shape}")
        combined_cam -= np.min(combined_cam)
        combined_cam /= np.max(combined_cam)
        return combined_cam

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release()


class GenerateCam:
    def __init__(self, model: object, input_item: Dict, target_layers: List) -> None:
        self.model = model
        self.input = input_item
        self.input_image = np.transpose(self.input['image'].data.numpy(), [1, 2, 0]) # (H, W, C)
        self.target_layers = target_layers

    def generate_cam(self):
        grad_cam = GradcamMultilayer(
            self.model,
            self.input_image,
            self.target_layers)
        with grad_cam as cam:
            cams = {}
            model = self.model
            model.eval()
            predictions = model.inference([self.input])
            instances = predictions[0]['instances'].to('cpu')
            pred_classes = [cls.item() for cls in instances.pred_classes]
            if len(pred_classes) == 0: # Return emtpy objects if no predictions
                return cams, instances
            unique_classes = np.unique(pred_classes)
            scores = predictions[0]['instances'].scores
            #class_scores = []
            #for scr, cls in zip(scores, pred_classes):
            #    if cls == unique_classes:
            #        class_scores.append(scr)
            #top_score = torch.max(torch.stack(class_scores))
            cams = {}
            for i in unique_classes:
                class_scores = [scr for scr, cls in zip(scores, pred_classes) if cls == i]
                top_score = torch.max(torch.stack(class_scores))
                top_score.backward(retain_graph=True)
                raw_cam = cam.get_cam()     
                cams[i] = (raw_cam)     
        return cams, instances

    def visualize_cam(self, raw_cam: np.ndarray):
        """ Generate heatmap from cam and resize it to input size
        Args
            raw_cam (ndarray): [H, W]
        Returns
            cam (ndarray)
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * raw_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        cam = 0.5*heatmap + 0.5*self.input_image
        cam -= np.max(np.min(cam), 0)
        cam /= np.max(cam)
        return cam

    def __call__(self) -> dict:
        """
        Returns
            final_cams (dict): keys are the cls and values are each of their corresponding cam image (ndarray)
            instances (object): a dectron2.structures.Instances object;
                                behaves like a dictionary
                                see: https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
        """
        raw_cams, instances = self.generate_cam()
        final_cams = {}
        for k, v in raw_cams.items():
            final_cams[k] = self.visualize_cam(v)
        return final_cams, instances
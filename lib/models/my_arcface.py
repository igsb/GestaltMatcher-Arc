import os

import onnx
import torch
import torch.nn as nn

from lib.models.utils import freeze_modules, unfreeze_modules
from onnx2torch import convert
import torch.nn.functional as F


class MyArcFace(nn.Module):

    def __init__(self,
                 num_classes,
                 dataset_base=os.path.join("saved_models", "glint360k_r50.onnx"),
                 device="cuda",
                 freeze=False):
        super(MyArcFace, self).__init__()

        # Load pretrained model from path if given (and not 'imagenet')
        if dataset_base:
            print(f"Loading pretrained weights from {dataset_base}")
            onnx_model = onnx.load(dataset_base)
            self.base = convert(onnx_model).to(device)
        else:
            print("We need to use a pretrained model for our MyArcFace model ...")
            exit()

        # magic values / layer numbers from onnx-file
        self.flatten_layer_number_mv = "252" if "r100" in dataset_base else "127"
        self.linear_layer_number_mv = "253" if "r100" in dataset_base else "128"
        self.BN_layer_number_mv = "254" if "r100" in dataset_base else "129"

        # Create a seperate feature group to allow easy addition of L2 regularization
        ## OLD
        # self.mv = "50" if "r100" in dataset_base else "25"  #magic value for the batchnorm number
        # self.features = nn.Sequential()
        # self.features.add_module('Gemm_0', self.base.get_submodule("Gemm_0"))
        # self.features.add_module(f'BatchNormalization_{self.mv}', self.base.get_submodule(f'BatchNormalization_{self.mv}')) #25 for r50, 50 for r100
        ## NEW
        self.features = nn.Sequential()
        self.features.add_module(f"Gemm", self.base.get_submodule(f"Gemm_{self.linear_layer_number_mv}"))
        self.features.add_module(f"BatchNormalization_{self.BN_layer_number_mv}", self.base.get_submodule(f"BatchNormalization_{self.BN_layer_number_mv}"))

        # If we want to split base and features we need to apply some magic ...
        self.transform_graph()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

        # freeze base model
        if freeze:
            print("Freezing model weights")
            freeze_modules(self.base)
        else:
            # Unfreezing all conv layer
            unfreeze_modules(self.base)

        # We always want to update our feature layer, otherwise we won't learn any features for clustering
        self.features.Gemm.requires_grad = True
        self.features.Gemm.weight.requires_grad = True

    def forward(self, x):
        x = self.base(x)
        if hasattr(self, 'features'):
            representations = self.features(x)
        else:
            representations = x
        y = self.classifier(representations)
        # y = self.classifier_hpo(representations)
        return y, representations

    def transform_graph(self):
        # Create new output: flatten_0 (we split the feature layer)
        for node in self.base.graph.nodes:
            # if node.name == 'flatten_0':
            if node.name == f"flatten_{self.flatten_layer_number_mv}":
                with self.base.graph.inserting_after(node):
                    self.base.graph.output(node)

        # Remove the original output, BN and feature layers
        self.base.graph.erase_node(list(self.base.graph.nodes)[::-1][0])
        self.base.graph.erase_node(list(self.base.graph.nodes)[::-1][0])
        self.base.graph.erase_node(list(self.base.graph.nodes)[::-1][0])

        # Make sure nothing is wrong and recompile the graph
        self.base.graph.lint()
        self.base.recompile()

        # self.base.delete_submodule("Gemm_0")
        # self.base.delete_submodule(f"BatchNormalization_{self.mv}") #25 for r50, 50 for r100
        self.base.delete_submodule(f"Gemm_{self.linear_layer_number_mv}")
        self.base.delete_submodule(f"BatchNormalization_{self.BN_layer_number_mv}")  # 129 for r50, 254 for r100
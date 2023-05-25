# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import torch

from models.backbone import Backbone, Joiner, TimmBackbone
from models.mdetr import MDETR
from models.position_encoding import PositionEmbeddingSine
from models.postprocessors import PostProcess, PostProcessSegm
from models.segmentation import DETRsegm
from models.transformer import Transformer

from models import build_model

dependencies = ["torch", "torchvision"]


def base_model():
    """
    Our base model with pre-trained Res50, RoBERTa-base, trained on Flickr-30k entities.
    """

    model_checkpoint = torch.hub.load_state_dict_from_url(
        url="https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth",
        map_location="cpu",
        check_hash=True,
    )
    model = build_model(model_checkpoint['args'])
    model.load_state_dict(model_checkpoint["weights"])
    return model
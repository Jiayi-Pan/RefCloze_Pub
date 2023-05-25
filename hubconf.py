import torch
from models import build_model
from utils.inference_utils import img_transform, post_processor

dependencies = ["torch", "torchvision"]

def base_model():
    """
    Our base model initialized from ResNet 50 and RoBERTa-base, pre-trained on Flickr-30k entities.
    """
    model_checkpoint = torch.hub.load_state_dict_from_url(
        url="https://github.com/Jiayi-Pan/temp/releases/download/Model/plain_model.pth",
        map_location="cpu",
        check_hash=True)
    model = build_model(model_checkpoint['args'])
    model.load_state_dict(model_checkpoint["weights"])
    return model, img_transform, post_processor

import torch
from PIL import Image
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

from matplotlib import patches,  lines
from matplotlib.patches import Polygon

img_transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def post_processor(outputs):
    # keep only predictions with 0.7+ confidence
    probs = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probs > 0.7).cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

    # Extract the text spans predicted by each box
    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
      item, pos = tok
      if pos < 255:
          # print()
          span = memory_cache["tokenized"].token_to_chars(0, pos)
          predicted_spans [item] += " " + caption[span.start:span.end]

    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]

    # unmasking
    mask_token_index = (memory_cache["tokenized"]['input_ids'] == model.transformer.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)
    print(mask_token_index)
    predicted_token_id = outputs['mlm_logits'][0, mask_token_index].argmax(axis=-1)
    predicted_token = model.transformer.tokenizer.decode(predicted_token_id)
    print(predicted_token)

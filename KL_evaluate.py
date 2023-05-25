import argparse
import copy
import json
import math
import os
from collections import defaultdict
from itertools import chain, repeat

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import (BatchSampler, DataLoader, Dataset,
							  SequentialSampler)
from torch.utils.data.sampler import BatchSampler, Sampler
from torchvision.ops import box_iou
from tqdm import tqdm
from yaml import parse

from models import build_model
from copy import deepcopy

import code

transform = T.Compose([
	T.Resize(800),
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def merge_dict(MainDict, AddedDict):
	for key in AddedDict:
		MainDict[key] += AddedDict[key]

def create_heat_map(img_size, bboxs, weights):
	hmap = torch.zeros(img_size[1], img_size[0]).double().cuda()
	weights = weights.double().cuda()
	assert bboxs.shape[0] == weights.shape[0]
	normed_weights = weights/sum(weights)
	for this_box, this_weight in zip(bboxs, normed_weights):
		x1,y1,x2,y2 = [int(e) for e in this_box.tolist()]
		cov = (x2-x1)*(y2-y1)
		if cov <= 0:
			continue
		hmap[y1:y2, x1:x2] += this_weight/((x2-x1)*(y2-y1))
	if torch.sum(hmap) == 0:
		return torch.ones_like(hmap)/(hmap.shape[0]*hmap.shape[1])
	hmap /= torch.sum(hmap)
	return hmap

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
	x_c, y_c, w, h = x.unbind(1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
		 (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return torch.stack(b, dim=1)

def box_coco_to_xyxy(bbox):
	[x, y, w, h] = bbox
	return [x,y,x+w, y+h]
	
def rescale_bboxes(out_bbox, size):
	img_w, img_h = size
	b = box_cxcywh_to_xyxy(out_bbox)
	b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
	return b


def mean(ls):
	return sum(ls)/len(ls)

def geo_mean(ls):
	return math.exp(sum([math.log(i) for i in ls])/len(ls))

def calc_matching_score_ANY_PROTOCOL(GT_bboxs, interested_bboxs):
	ious = box_iou(interested_bboxs, GT_bboxs)
	any_results = []
	correct = 0
	for ious_for_this in ious:
		any_results.append(float(ious_for_this.max()))
	return any_results

def merge_bboxs(bboxs):
	tensor_boxs = bboxs # (N, 4)
	xmin = tensor_boxs[:, 0].min()
	ymin = tensor_boxs[:, 1].min()
	xmax = tensor_boxs[:, 2].max()
	ymax = tensor_boxs[:, 3].max()
	return torch.tensor([xmin, ymin, xmax, ymax], device=device)

def calc_matching_score_ALL_PROTOCOL(GT_bboxs, interested_bboxs):
	if len(interested_bboxs) == 0:
		return 0
	GT_big_box = merge_bboxs(GT_bboxs)
	Interested_big_box = merge_bboxs(interested_bboxs)
	# print(GT_big_box)
	# print(Interested_big_box)
	return box_iou(Interested_big_box[None], GT_big_box[None]).float()

class LG_Dataset(Dataset):
	def __init__(self, annotation, flickr_img_dir, coco_img_dir, gqa_img_dir):
		if type(annotation) == str:
			with open(annotation, 'r') as f:
				self.data = json.load(f)
		else:
			self.data = annotation
		self.flickr_img_dir = flickr_img_dir
		self.coco_img_dir = coco_img_dir
		self.gqa_img_dir = gqa_img_dir
		self.transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

	def __len__(self):
		return len(self.data['tests'])
	
	def get_height_and_width(self, idx):
		this_info =  self.data['grounding_pairs'][self.data['tests'][idx]['pair_id']]['data']['img_info']
		return this_info['height'], this_info['width']

	def __getitem__(self, idx):
		# lang-part
		pair_id = self.data['tests'][idx]['pair_id']
		pair_info = self.data['grounding_pairs'][pair_id]
		assert pair_id == pair_info['pair_id']
		pair_data = pair_info['data']
		## captipon
		caption = pair_data['img_info']['caption']
		## mask_region
		mask_regions = self.data['tests'][idx]['mask_regions']
		token_idxs = [reg[0] for reg in mask_regions]
		word_spans = [pair_data['parse_result'][i][2] for i in token_idxs]

		# image-loading
		img_file_name = pair_data['img_info']['file_name']
		dataset_name = pair_data['img_info']['dataset_name']
		if self.flickr_img_dir == None:
			# for easier debugging
			img = None
			img_path = None
		else:
			if dataset_name == "flickr":
				img_path = self.flickr_img_dir + img_file_name
			elif dataset_name in ["refcoco", "refcoco+", "refcocog"]:
				img_path = self.coco_img_dir + img_file_name
			elif dataset_name == "gqa" or dataset_name == "VG":
				img_path = self.gqa_img_dir + img_file_name
			else:
				raise 'Unknown dataset name'

			# print(img_path)
			# print(np.asarray(img).shape)
			img = Image.open(img_path).convert('RGB')
			img_hw = img.size
			img = self.transform(img)

		## related bbox
		bboxs = []
		phrase_region_spans = []
		for word_span in word_spans:
			this_boxs = []
			for a in pair_data['annotations']:
				this_spans = a['bbox_info']['tokens_positive']
				is_in = False
				for span in this_spans:
					if word_span[0] >= span[0] and word_span[1] <= span[1]:
						is_in = True
						break
				if is_in:
					phrase_region_spans = a['bbox_info']['tokens_positive']
					this_boxs.append(a['bbox_info']['bbox'])
			# one grounding phrase must have at least one bbox
			assert len(this_boxs) > 0, (idx, pair_data)
			
			bboxs.append(this_boxs)

		return { 'img': img, 'text_info':{'img_size': img_hw, 'data_id': idx, 'caption':caption, 'mask_regions': mask_regions, 'word_spans': word_spans, "phrase_spans": phrase_region_spans}, 'bboxs':bboxs, 'dataset_name': dataset_name}

def mask_and_tokenize(caption_batch, word_spans_batch, tokenizer):
	encs = tokenizer(caption_batch, return_tensors='pt')
	# initially all masked
	labels = torch.ones_like(encs.input_ids)*(-100)
	GTs = []
	for batch_idx, word_spans in enumerate(word_spans_batch):
		# word_spans is [[start, end], [start, end], ...]
		for word_span in word_spans:
			# find [start, end]

			# start token index
			start_token_idx = None
			start_ctr = word_span[0]
			while start_token_idx == None:
				assert start_ctr >= 0
				start_token_idx = encs.char_to_token(batch_idx, start_ctr)
				start_ctr -= 1

			# end token index
			end_token_idx = None
			end_ctr = word_span[1]-1
			while end_token_idx == None:
				assert end_ctr <= len(caption_batch[batch_idx]), (caption_batch, word_spans_batch)
				end_token_idx = encs.char_to_token(batch_idx, end_ctr)
				end_ctr += 1
			end_token_idx += 1 # end_token_idx is exclusive

			# applying mask, construct annotations
			for token_idx in range(start_token_idx, end_token_idx):
				if encs.input_ids[batch_idx, token_idx] != tokenizer.mask_token_id:
					labels[batch_idx,token_idx] = encs.input_ids[batch_idx, token_idx]
			encs.input_ids[batch_idx, start_token_idx:end_token_idx] = tokenizer.mask_token_id
	return encs, labels

class GroupedBatchSampler(BatchSampler):
	"""
	NOTE: Modified from torchvision. Instead of leaving remaining ones for efficiency, this implementation will run through the entire dataset once, when not enough data to form a full batch, the remaining data will be used to form a smaller batch.

	Wraps another sampler to yield a mini-batch of indices.
	It enforces that the batch only contain elements from the same group.
	It also tries to provide mini-batches which follows an ordering which is
	as close as possible to the ordering from the original sampler.
	Args:
		sampler (Sampler): Base sampler.
		group_ids (list[int]): If the sampler produces indices in range [0, N),
			`group_ids` must be a list of `N` ints which contains the group id of each sample.
			The group ids must be a continuous set of integers starting from
			0, i.e. they must be in the range [0, num_groups).
		batch_size (int): Size of mini-batch.
	"""

	def __init__(self, sampler, group_ids, batch_size):
		if not isinstance(sampler, Sampler):
			raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
		self.sampler = sampler
		self.group_ids = group_ids
		self.batch_size = batch_size

	def __iter__(self):
		buffer_per_group = defaultdict(list)
		samples_per_group = defaultdict(list)

		# batch with full batch-size 
		num_batches = 0
		for idx in self.sampler:
			group_id = self.group_ids[idx]
			buffer_per_group[group_id].append(idx)
			samples_per_group[group_id].append(idx)
			if len(buffer_per_group[group_id]) == self.batch_size:
				yield buffer_per_group[group_id]
				num_batches += 1
				del buffer_per_group[group_id]
			assert len(buffer_per_group[group_id]) < self.batch_size

		# batch with remaining samples
		for k, v in buffer_per_group.items():
			if len(v) > 0:
				yield v

def _compute_aspect_ratios_slow(dataset, indices=None):
	print(
		"Your dataset doesn't support the fast path for "
		"computing the aspect ratios, so will iterate over "
		"the full dataset and load every image instead. "
		"This might take some time..."
	)
	if indices is None:
		indices = range(len(dataset))

	class SubsetSampler(Sampler):
		def __init__(self, indices):
			self.indices = indices

		def __iter__(self):
			return iter(self.indices)

		def __len__(self):
			return len(self.indices)

	sampler = SubsetSampler(indices)
	data_loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=1,
		sampler=sampler,
		num_workers=14,  # you might want to increase it for faster processing
		collate_fn=lambda x: x[0],
	)
	aspect_ratios = []
	with tqdm(total=len(dataset)) as pbar:
		for _i, (img, _) in enumerate(data_loader):
			pbar.update(1)
			height, width = img.shape[-2:]
			aspect_ratio = float(width) / float(height)
			aspect_ratios.append(aspect_ratio)
	return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
	if indices is None:
		indices = range(len(dataset))
	aspect_ratios = []
	for i in indices:
		height, width = dataset.get_height_and_width(i)
		aspect_ratio = float(width) / float(height)
		aspect_ratios.append(aspect_ratio)
	return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
	if indices is None:
		indices = range(len(dataset))
	aspect_ratios = []
	for i in indices:
		img_info = dataset.coco.imgs[dataset.ids[i]]
		aspect_ratio = float(img_info["width"]) / float(img_info["height"])
		aspect_ratios.append(aspect_ratio)
	return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
	if indices is None:
		indices = range(len(dataset))
	aspect_ratios = []
	for i in indices:
		# this doesn't load the data into memory, because PIL loads it lazily
		width, height = Image.open(dataset.images[i]).size
		aspect_ratio = float(width) / float(height)
		aspect_ratios.append(aspect_ratio)
	return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
	if indices is None:
		indices = range(len(dataset))

	ds_indices = [dataset.indices[i] for i in indices]
	return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
	if hasattr(dataset, "get_height_and_width"):
		return _compute_aspect_ratios_custom_dataset(dataset, indices)

	if isinstance(dataset, torchvision.datasets.CocoDetection):
		return _compute_aspect_ratios_coco_dataset(dataset, indices)

	if isinstance(dataset, torchvision.datasets.VOCDetection):
		return _compute_aspect_ratios_voc_dataset(dataset, indices)

	if isinstance(dataset, torch.utils.data.Subset):
		return _compute_aspect_ratios_subset_dataset(dataset, indices)

	# slow path
	return _compute_aspect_ratios_slow(dataset, indices)

def _quantize(x, bins):
	bins = copy.deepcopy(bins)
	bins = sorted(bins)
	quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
	return quantized

def create_aspect_ratio_groups(dataset, k=0):
	aspect_ratios = compute_aspect_ratios(dataset)
	bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
	groups = _quantize(aspect_ratios, bins)
	# count number of elements per group
	counts = np.unique(groups, return_counts=True)[1]
	fbins = [0] + bins + [np.inf]
	print(f"Using {fbins} as bins for aspect ratio quantization")
	print(f"Count of instances per bin: {counts}")
	return groups

def calc_group_ids(dataset_in):
	ratios_seq = _compute_aspect_ratios_custom_dataset(dataset_in)
	temp_ratios_id_dict = dict(enumerate(list(set(ratios_seq))))
	ratio_id_dict = dict([(value, key) for key, value in temp_ratios_id_dict.items()])
	ids = []
	for r in ratios_seq:
		ids.append(ratio_id_dict[r])
	return ids

def build_aspect_ratio_grouped_sampler(dataset_in, batch_size):
	base_sampler = SequentialSampler(range(len(dataset_in)))
	group_ids = calc_group_ids(dataset_in)
	return GroupedBatchSampler(base_sampler, group_ids, batch_size)

def collate_batch(batch):
	imgs = torch.stack([a['img'] for a in batch])
	infos = [{'text_info': a['text_info'], 'bboxs': a['bboxs'], 'dataset_name': a['dataset_name']} for a in batch]
	return imgs, infos

def build_dataloader(dataset, batch_size, num_workers):
	s = build_aspect_ratio_grouped_sampler(dataset, batch_size)
	return   DataLoader(dataset, batch_sampler=s, collate_fn=collate_batch, num_workers = num_workers, pin_memory=True)

def top10_probs(mask_region_probs, mask_idxs, input_ids, tokenizer):
	topK_in_probs = torch.topk(mask_region_probs, k=10, dim=-1)

	# topK in at each token, value and index 
	V = topK_in_probs.values
	I = topK_in_probs.indices

	acc_mat = V[0,:]
	for i in range(1, V.shape[0]):
		acc_mat = acc_mat.unsqueeze(-1)
		acc_mat @= V[i,:].unsqueeze(0)
	
	topK_result = torch.topk(acc_mat.flatten(), k=10)
	topK_values = topK_result.values
	topK_idxs = topK_result.indices

	idxs_result = []
	all_num = acc_mat.numel()
	prod = all_num 
	idx = 1
	size = acc_mat.shape[0]
	for i in range(len(acc_mat.shape)):
		# 1000, 100, 10
		this_idx = torch.div(topK_idxs % prod, int(prod / size), rounding_mode="floor")
		idxs_result.append(this_idx.int())
		prod /= size
	
	# return 
	ret = []
	for i in range(10):
		this_r = {'rank': i+1, 'prob': topK_values[i].item()}
		this_r['tokens'] = []
		this_r['tokens_ids'] = []
		this_r['tokens_rel_rank'] = []
		for j in range(len(idxs_result)):
			tk_idx = I[j, idxs_result[j][i]].item()
			this_r['tokens'].append(tokenizer.decode(tk_idx))
			this_r['tokens_ids'].append(tk_idx)
			this_r['tokens_rel_rank'].append(idxs_result[j][i].item())
		this_r['tokens_ids'] = tuple(this_r['tokens_ids'])
		this_r['tokens_rel_rank'] = tuple(this_r['tokens_rel_rank'])
		rev_input_ids = input_ids.clone()
		for idx, i in enumerate(mask_idxs):
			rev_input_ids[i] = this_r['tokens_ids'][idx]
		this_r['rev_cap'] = tokenizer.decode(rev_input_ids, skip_special_tokens=True)
		ret.append(this_r)
	return ret

def resize_phrase_span_by_mask(regions, mask_region, tknz):
	resized_regions = sorted(regions, key=lambda x: x[0])
	start_resize = False
	resize_to = None
	for reg in resized_regions:
		if start_resize:
			reg[0] += resize_to
			reg[1] += resize_to
		if mask_region[0] >= reg[0] and mask_region[1] <= reg[1]:
			# mask is in this region
			start_resize = True
			resize_to = len(tknz.mask_token) - (mask_region[1] - mask_region[0])
			reg[1] += resize_to
	return resized_regions

def mask_region_to_token_idxs(encs, phrase_region, batch_idx=0):
	tkidxs = set()
	for reg in phrase_region:
		for char_idx in range(reg[0], reg[1]):
			id = encs.char_to_token(batch_idx, char_idx)
			if id != None:
				tkidxs.add(id)
	return tkidxs

def pos_idxs_to_distribution(pos_idxs, dim=256, device='cuda'):
	pos_idxs = list(pos_idxs)
	t = torch.zeros(dim, device=device)
	t[pos_idxs] = 1/len(pos_idxs)
	return t

def math_KL_div(P, Q):
	assert len(P.shape) == len(Q.shape)
	assert len(P.shape) == 1
	kl = 0.
	for idx, p in enumerate(P):
		q = Q[idx]
		if q <= 0.:
			return float('inf')
		if p <= 1e-4:
			continue
		kl += p * torch.log(p / q)
	return kl

def eval_one_batch(model, batch_in, device='cuda', amp=False):
	Batch_Result = []
	Word_Embedding_Dic = defaultdict(list)
	if hasattr(model, "module"):
		tknz = model.module.transformer.tokenizer
	else:
		tknz = model.transformer.tokenizer
	imgs, infos = batch_in
	imgs = imgs.to(device)
	recovered_caps = []
	labels = []
	dp_ids = []
	batch_phrase_spans_after_mask = []
	for dp in infos:
		cap = dp['text_info']['caption']
		dp_ids.append(dp['text_info']['data_id'])
		word_spans = dp['text_info']['word_spans']
		POS = dp['text_info']['mask_regions'][0][2]
		dataset_name = dp['dataset_name']
		encs, label = mask_and_tokenize([cap], [word_spans], tknz)
		labels.append(label)
		recovered_cap = tknz.decode(encs['input_ids'][0][1:-1])
		recovered_caps.append(recovered_cap)

		# md metric
		phrase_spans = dp['text_info']['phrase_spans']
		mask_spans = dp['text_info']['word_spans'][0] # only one mask
		phrase_spans_after_mask = resize_phrase_span_by_mask(phrase_spans, mask_spans, tknz)
		for span_idx in phrase_spans_after_mask:
			if span_idx[0] < 0 or span_idx[1] < 0:
				code.interact(local=locals())
		batch_phrase_spans_after_mask.append(phrase_spans_after_mask)

	encs = tknz(recovered_caps, return_tensors='pt', padding=True)
	with torch.cuda.amp.autocast(amp):
		with torch.no_grad():
			memory_cache = model(imgs, encs, encode_and_save=True)
			model_outputs = model(imgs, encs, encode_and_save=False, memory_cache=memory_cache)
	for dp_idx in range(len(infos)):
		## Find interested token indexes
		zero_label = labels[dp_idx].clone()
		zero_label[labels[dp_idx]==-100] =0 
		interested_idxs = [int(pos[1]) for pos in zero_label.nonzero()]
		masked_token_gt_idxs = [zero_label[idx[0], idx[1]] for idx in zero_label.nonzero()]

		## Find interested proposal indexes
		interested_proposal_idxs = []

		## Calc PPL on mask tokens
		mask_region_logits = model_outputs['mlm_logits'].float()[dp_idx,interested_idxs]
		mask_region_probs = torch.nn.functional.softmax(mask_region_logits, dim=-1)
		mask_region_GT_probs = mask_region_logits.softmax(-1)[0, masked_token_gt_idxs]
		# ppl
		ppl_score =  math.exp(sum([math.log(1/p) for p in mask_region_GT_probs])/len(mask_region_GT_probs))

		## TOP10 infos
		top10_result = top10_probs(mask_region_probs, interested_idxs, encs.input_ids[dp_idx], tknz)

		this_raw = {}
		hit_at_x = None
		this_raw['ppl'] = float(ppl_score) 
		this_raw['num_token'] = len(interested_idxs)
		this_raw['rank'] = hit_at_x
		this_raw['top10'] = top10_result
		this_raw['raw_logits'] = mask_region_logits.cpu()
		Batch_Result.append(this_raw)
	return Batch_Result, Word_Embedding_Dic


def run_one_model(model, dataloader, meta, file_name):
	RAWs = []
	Word_Embedding_Dic = defaultdict(list)
	for idx, batch_data in enumerate(tqdm(dataloader)):
		raw_batch, emb_dict = eval_one_batch(model, batch_data, amp=True)
		merge_dict(Word_Embedding_Dic, emb_dict)
		RAWs += raw_batch
	Word_Embeddings = {k: np.mean(v, axis=0) for k, v in Word_Embedding_Dic.items()}
	Result = {"meta":meta, "is_two_stage": False, "raw": RAWs, "embeddings": Word_Embeddings}
	torch.save(Result, file_name)

if __name__ == '__main__':
	print("Start Evaluation")
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	device = 'cuda'

	parser = argparse.ArgumentParser(description="Evaluate Our Model on Unigram")
	parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
	parser.add_argument('--log_path', type=str, default=None, help='Path to the log file')
	parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset')

	args = parser.parse_args()

	ckpt = torch.load(args.model_path)
	model, criterion, contrastive_criterion, qa_criterion, mvm_criterion, mlm_criterion, weight_dict = build_model(ckpt['args'])
	model.load_state_dict(ckpt['model'])
	_ = model.eval().cuda()

	# with open('/nfs/turbo/coe-chaijy/jiayipan/Vision-Language-Project/EMNLP/Evaluation/dataset/Filtered_Dataset.json', 'r') as f:
	# with open('/nfs/turbo/coe-chaijy/jiayipan/Vision-Language-Project/EMNLP/Evaluation/dataset/Flickr_Dataset.json', 'r') as f:
	with open(args.dataset_path , 'r') as f:
		data = json.load(f)
	print(f"Num of Tests: {len(data['tests'])}")

	dataset = LG_Dataset(data, "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/pretrain_data/flickr30k-images/", "/nfs/turbo/coe-chaijy/datasets/mscoco-2014/all/", "/scratch/chaijy_root/chaijy2/jiayipan/datasets/GQA/images/")
	dataloader = build_dataloader(dataset, 32, num_workers=4)
	meta = f"model: {args.model_path}"
	run_one_model(model, dataloader, meta, args.log_path)
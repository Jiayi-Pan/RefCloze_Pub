from distutils.command.build import build
from tabnanny import verbose
import torch
import numpy as np
import os
import pprint
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

pp = pprint.PrettyPrinter(indent=4)

def geometric_mean(x): 
    a = np.log(x)
    return np.exp(a.mean())

def mean(x):
    if len(x) == 0:
        return 0
    return sum(x)/len(x)
import json

# POS_json = "slicing_uni/coco-flickr/POS.json"
# Freq_json = "slicing_uni/coco-flickr/Freq.json"
# Word_json = "slicing_uni/ana/Word.json"
from random import choices

# with open(POS_json, "r") as f:
#     POS_ids_data = json.load(f)
# with open(Freq_json, "r") as f:
#     Freq_ids_data = json.load(f)

class Slice():
    def __init__(self, name):
        self.name = name
        self.dp_ids = []
    
    def add_datapoint(self, idx):
        self.dp_ids.append(idx)
    
    def __add__(self, other):
        # intersection
        assert type(other) is Slice
        name = f"[Intersection: [{self.name}] and [{other.name}]]"
        dp_ids = list(set(self.dp_ids) & set(other.dp_ids))
        NewSlice = Slice(name)
        NewSlice.dp_ids = dp_ids
        return NewSlice

    def __mul__(self, other):
        # union
        assert type(other) is Slice
        name = f"[Union: [{self.name}] and [{other.name}]]"
        dp_ids = list(set(self.dp_ids).union(set(other.dp_ids)))
        NewSlice = Slice(name)
        NewSlice.dp_ids = dp_ids
        return NewSlice
    
    def __str__(self) -> str:
        return f"{self.name} \n \t data num: {len(self.dp_ids)} \t top5: {self.dp_ids[:5]}"

class SliceManager():
    def __init__(self, eval_dataset_name):
        if eval_dataset_name == "flickr-unseen":
            self.eval_dataset_name = eval_dataset_name
            word_json = "slice_db/unseen/Word.json"
            word_list = "slice_db/unseen/WordList.json"
            with open(word_json, "r") as f:
                self.word_ids_data = json.load(f)
            with open(word_list, "r") as f:
                self.word_list = json.load(f)
            

        elif eval_dataset_name == "flickr-eval":
            self.eval_dataset_name = eval_dataset_name
            word_json = "slice_db/eval/Word.json"
            word_list = "slice_db/eval/WordList.json"
            with open(word_json, "r") as f:
                self.word_ids_data = json.load(f)
            with open(word_list, "r") as f:
                self.word_list = json.load(f)
        else:
            assert False, "Unregistered dataset name!"

    def build_dataset_slice(self, dataset_name):
        this_test = Slice(dataset_name)
        return this_test

    def build_word_slice(self, word):
        Word_Test = Slice(f"{word}")
        if word in self.word_ids_data:
            Word_Test.dp_ids = self.word_ids_data[word]
        else:
            Word_Test.dp_ids = []
        return Word_Test

def eval_one_result(result, slicings, verbose = False):
    if type(result) is str:
        result = torch.load(result, map_location=torch.device('cpu'))

    if verbose:
        print("==========================")
        print(f"{result['meta']}") 
    
    sorted_raw = sorted(result['raw'], key=lambda x: x['dpid'])
    for i in range(len(sorted_raw)):
        assert sorted_raw[i]['dpid'] == i

    Stats = []
    for slicing in slicings: 
        this_stats = {}
        this_slice_raw = [sorted_raw[i] for i in slicing.dp_ids]
        ppls = []
        any_ious = []
        all_ious = []
        soft_match_l1s = []
        soft_match_l2s = []
        soft_match_kls = []
        ranks = []
        bridge = {"raw_top1": {"normed_probs":[], "kls":[], "any_ious":[]}, "proposed": {"normed_probs":[], "kls":[], "any_ious":[]}}
        for dp in this_slice_raw:
            ppls.append(dp["ppl"])
            any_ious.append(dp["any_iou"])
            all_ious.append(dp["all_iou"])
            if dp['rank'] is not None:
                ranks.append(dp['rank'])
        ranks = np.asarray(ranks) 
        this_stats["ppl"] = geometric_mean(ppls)
        def ge_50(x):
            if type(x) is list:
                x = np.asarray(x)
            return np.sum(x >= 0.50) / x.shape[0]
        this_stats["all_iou"] = mean(all_ious)
        this_stats["any_iou"] = mean([mean(this_ious) for this_ious in any_ious])
        # All top 1 hits (>0.5 for bboxs)
        lang_hit = 0
        all_box_hit = 0
        all_hit = 0
        any_box_hit = 0
        any_hit = 0
        all_num = 0
        for r in this_slice_raw:
            all_num += 1
            prod = (r['ppl'] * r['top10'][0]['prob']) 
            diff = (1/r['ppl'] - r['top10'][0]['prob']) 
            is_lang_hit = False
            if prod <= 1.0001 and prod >= 0.9999:
                lang_hit += 1
                is_lang_hit = True
            if r['all_iou'] >= 0.5:
                all_box_hit += 1
                if is_lang_hit:
                    all_hit += 1
            if mean(r['any_iou']) >= 0.5:
                any_box_hit += 1
                if is_lang_hit:
                    any_hit += 1
        this_stats['mask_hit'] =  lang_hit / all_num
        this_stats["all_box_hit"] = all_box_hit / all_num
        this_stats["all_hit"] = all_hit / all_num
        this_stats["any_box_hit"] = any_box_hit / all_num
        this_stats["any_hit"] = any_hit / all_num

        # these are just single word hit freq
        this_stats["hit_top1"] = mean(ranks <= 1)
        this_stats["hit_top3"] = mean(ranks <= 3)
        this_stats["hit_top5"] = mean(ranks <= 5)
        this_stats["hit_top10"] = mean(ranks <= 10)
        this_stats['embedding'] = result['embeddings'][slicing.name]
        Stats.append(this_stats)
    return Stats

def AOA_eval(result_path, slicings, threads=16):
    all_result_files = os.listdir(result_path)
    print(f"{len(all_result_files)} results found")
    Stats = {}
    for fn in all_result_files:
        assert fn[:15] == "checkpoint_step"
    full_paths = [os.path.join(result_path, f) for f in all_result_files]

    # Parallelize loading/evaluating
    with mp.Pool(threads) as p:
        results = list(tqdm(p.imap(partial(eval_one_result, slicings=slicings, verbose=False), full_paths), total=len(full_paths)))
    # results = [eval_one_result(f, slicings=slicings, verbose=False) for f in full_paths]
    for idx, fn in enumerate(all_result_files):
        assert fn[:15] == "checkpoint_step"
        step_num = int(fn[15:-4])
        Stats[step_num] = {"file_path": os.path.join(result_path, fn)}
        Stats[step_num]["result"] = results[idx]

    # return results
    Stats = dict(sorted(Stats.items()))
    ResultSummary = {}
    stat_names = ["ppl", "all_iou", "any_iou", "hit_top1", "hit_top3", "hit_top5", "hit_top10", "mask_hit", "all_box_hit", "all_hit", "any_box_hit", "any_hit", "embedding"]
    for test_idx in range(len(slicings)):
        ResultSummary[slicings[test_idx].name] = {name: [] for name in stat_names}
        ResultSummary[slicings[test_idx].name]['step'] = []
        for k, v in Stats.items():
            ResultSummary[slicings[test_idx].name]["step"].append(k)
            for stat_name in stat_names:
                ResultSummary[slicings[test_idx].name][stat_name].append(v['result'][test_idx][stat_name])
        ResultSummary = {}
    stat_names = ["ppl", "all_iou", "any_iou", "g_score_all", "g_score_any", "bridge_raw_top1_norm_prob", "bridge_raw_top1_kl", "bridge_raw_top1_any_iou", "bridge_raw_top1_any_box_hit","bridge_proposed_norm_prob", "bridge_proposed_kl", "bridge_proposed_any_iou", "mask_hit", "all_box_hit", "all_hit", "any_box_hit", "any_hit", "soft_match_l1", "soft_match_l2", "soft_match_kl", "all_hit_top3", "all_hit_top5", "all_hit_top10", "any_hit_top3", "any_hit_top5", "any_hit_top10", "word_dis", "amb_any_iou", "amb_any_box_hit", "mask_hit_top3", "mask_hit_top5", "mask_hit_top10"]
    for test_idx in range(len(slicings)):
        ResultSummary[slicings[test_idx].name] = {name: [] for name in stat_names}
        ResultSummary[slicings[test_idx].name]['step'] = []
        for k, v in Stats.items():
            ResultSummary[slicings[test_idx].name]["step"].append(k)
            for stat_name in stat_names:
                ResultSummary[slicings[test_idx].name][stat_name].append(v['result'][test_idx][stat_name])
    return ResultSummary
from distutils.command.build import build
from tabnanny import verbose
import torch
import numpy as np
import os
import pprint
from tqdm import tqdm
from analysis_utils.stats_tools import *
from analysis_utils.slicing_utils import Slice_Manager
import multiprocessing as mp
from functools import partial
torch.multiprocessing.set_sharing_strategy('file_system')
pp = pprint.PrettyPrinter(indent=4)

from nltk.corpus import wordnet as wn

def word_similarity(w1, w2, method='wup'):
    w1_candidates = wn.synsets(w1)
    w2_candidates = wn.synsets(w2)
    max_similarity = 0
    for w1_candidate in w1_candidates:
        for w2_candidate in w2_candidates:
            if method == 'wup':
                similarity = w1_candidate.wup_similarity(w2_candidate)
            elif method == 'path':
                similarity = w1_candidate.path_similarity(w2_candidate)
            else:
                raise ValueError(f'Invalid similarity method: {method}')
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    return max_similarity

def calc_word_sim(dp, test):
    gt_word = test['mask_regions'][0][1]
    pred_word = ''.join(dp['top10'][0]['tokens'])
    pred_word = pred_word.replace(' ', '')
    word_sim_wup = word_similarity(gt_word, pred_word, method='wup')
    word_sim_path = word_similarity(gt_word, pred_word, method='path')
    return {'wup': word_sim_wup, 'path': word_sim_path}


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
        # ----- collect raw data -----
        this_stats = {}
        max_idx = max(slicing.dp_ids)
        if max_idx > len(sorted_raw):
            assert False, f"mismatched AOA & slicing {max_idx} > {len(sorted_raw)}"
        this_slice_raw = [sorted_raw[i] for i in slicing.dp_ids]
        ppls = []
        amb_any_ious = []
        amb_any_box_hits = []
        word_diss = []
        any_ious = []
        all_ious = []
        soft_match_l1s = []
        soft_match_l2s = []
        soft_match_kls = []
        bridge = {"raw_top1": {"normed_probs":[], "kls":[], "any_ious":[]}, "proposed": {"normed_probs":[], "kls":[], "any_ious":[]}}
        for dp, dp_id in zip(this_slice_raw, slicing.dp_ids):
            word_diss.append(dp['word_dis']) if 'word_dis' in dp else word_diss.append(-1)
            ppls.append(dp["ppl"])
            any_ious.append(dp["any_iou"])
            all_ious.append(dp["all_iou"])
            soft_match_l1s.append(dp["soft_match_l1"])
            soft_match_l2s.append(dp["soft_match_l2"])
            soft_match_kls.append(dp["soft_match_kl"])
            for stat_name in ["normed_prob", "kl", "any_iou"]:
                bridge["raw_top1"][stat_name + 's'].append(dp['grounding']["raw_top1"][stat_name])
            for stat_name in ["normed_prob", "kl", "any_iou"]:
                T = bridge["proposed"][stat_name + 's']
                for a in dp['grounding']['proposed']:
                    T.append(a[stat_name])
            if 'cat_exclusive_any_iou' in dp['grounding']['raw_top1']:
                if dp['grounding']['raw_top1']['cat_exclusive_any_iou'] is not None:
                    amb_any_ious.append(dp['grounding']['raw_top1']['cat_exclusive_any_iou'][0])
            # word sim
            # word_sim = calc_word_sim(dp,data['tests'][dp_id])
            # path_word_sims.append(word_sim['path'])
            # wup_word_sims.append(word_sim['wup'])
        # ----- process to stats -----
        this_stats["ppl"] = geometric_mean(ppls)
        this_stats["word_dis"] = mean(word_diss)
        this_stats["all_iou"] = mean(all_ious)
        this_stats["any_iou"] = mean([mean(ious) for ious in any_ious])
        this_stats["g_score_all"] = geometric_mean(ppls)/mean(all_ious) if mean(all_ious) > 0 else float('inf')
        this_stats["g_score_any"] = geometric_mean(ppls)/mean([mean(ious) for ious in any_ious]) if mean([mean(ious) for ious in any_ious]) > 0 else float('inf')

        # this is the grounding box matching score obtained by heat map 
        this_stats['soft_match_l1'] = mean(soft_match_l1s)
        this_stats['soft_match_l2'] = mean(soft_match_l2s)
        this_stats['soft_match_kl'] = mean(soft_match_kls)
        this_stats['amb_any_iou'] = mean(amb_any_ious)
        this_stats['amb_any_box_hit'] = mean([1 if x >= 0.5 else 0 for x in amb_any_ious])

        # All top 1 hits (>0.5 for bboxs)
        lang_hit = 0
        lang_hit_top3 = 0
        lang_hit_top5 = 0
        lang_hit_top10 = 0
        all_box_hit = 0
        all_hit = 0
        all_hit_top3 = 0
        all_hit_top5 = 0
        all_hit_top10 = 0
        any_box_hit = 0
        any_hit = 0
        any_hit_top3 = 0
        any_hit_top5 = 0
        any_hit_top10 = 0
        all_num = 0
        for r in this_slice_raw:
            all_num += 1
            diff = (1/r['ppl'] - r['top10'][0]['prob']) 
            is_lang_hit = False
            is_all_iou_hit = False
            is_any_iou_hit = False
            if diff <= 0.00001 and diff >= -0.00001:
                lang_hit += 1
                is_lang_hit = True
            if r['all_iou'] >= 0.5:
                all_box_hit += 1
                is_all_iou_hit = True
                if is_lang_hit:
                    all_hit += 1
            if mean(r['any_iou']) >= 0.5:
                any_box_hit += 1
                is_any_iou_hit = True
                if is_lang_hit:
                    any_hit += 1

            hit_top_i = None
            for top_i in range(10):
                # diff = (1/r['ppl'] - r['top10'][top_i]['prob']) 
                prod = r['ppl'] * r['top10'][top_i]['prob']
                if prod <= 1.000001 and prod >= 0.999999:
                    if top_i < 3:
                        lang_hit_top3 += 1
                        if is_any_iou_hit:
                            any_hit_top3 += 1
                        if is_all_iou_hit:
                            all_hit_top3 += 1
                    if top_i < 5:
                        lang_hit_top5 += 1
                        if is_any_iou_hit:
                            any_hit_top5 += 1
                        if is_all_iou_hit:
                            all_hit_top5 += 1
                    if top_i < 10:
                        lang_hit_top10 += 1
                        if is_any_iou_hit:
                            any_hit_top10 += 1
                        if is_all_iou_hit:
                            all_hit_top10 += 1
                    break
        this_stats['hit_top_i'] = hit_top_i
        this_stats['mask_hit'] =  lang_hit / all_num
        this_stats['mask_hit_top3'] =  lang_hit_top3 / all_num
        this_stats['mask_hit_top5'] =  lang_hit_top5 / all_num
        this_stats['mask_hit_top10'] =  lang_hit_top10 / all_num
        this_stats["all_box_hit"] = all_box_hit / all_num
        this_stats["all_hit"] = all_hit / all_num
        this_stats["all_hit_top3"] = all_hit_top3 / all_num
        this_stats["all_hit_top5"] = all_hit_top5 / all_num
        this_stats["all_hit_top10"] = all_hit_top10 / all_num
        this_stats["any_box_hit"] = any_box_hit / all_num
        this_stats["any_hit"] = any_hit / all_num
        this_stats["any_hit_top3"] = any_hit_top3 / all_num
        this_stats["any_hit_top5"] = any_hit_top5 / all_num
        this_stats["any_hit_top10"] = any_hit_top10 / all_num

        this_stats['bridge_raw_top1_norm_prob'] = mean(bridge['raw_top1']['normed_probs'])
        this_stats['bridge_raw_top1_kl'] = mean(bridge['raw_top1']['kls'])
        raw_top1_avg_any_iou = [mean(x) for x in bridge['raw_top1']['any_ious']]
        this_stats['bridge_raw_top1_any_iou'] = mean(raw_top1_avg_any_iou)
        this_stats['bridge_raw_top1_any_box_hit'] = mean([1 if x >= 0.5 else 0 for x in raw_top1_avg_any_iou])

        this_stats['bridge_proposed_norm_prob'] = mean(bridge['proposed']['normed_probs'])
        this_stats['bridge_proposed_kl'] = mean(bridge['proposed']['kls'])
        this_stats['bridge_proposed_any_iou'] = mean([mean(ious) for ious in bridge['proposed']['any_ious']])
        Stats.append(this_stats)
    return Stats

def AOA_eval(result_path, slicings):
    all_result_files = os.listdir(result_path)
    print(f"{len(all_result_files)} results found")
    Stats = {}
    for fn in all_result_files:
        assert fn[:15] == "checkpoint_step"
    full_paths = [os.path.join(result_path, f) for f in all_result_files]

    # Parallelize loading/evaluating
    with mp.Pool(8) as p:
        results = list(tqdm(p.imap(partial(eval_one_result, slicings=slicings, verbose=False), full_paths), total=len(full_paths)))
    # results = [eval_one_result(f, slicings=slicings, verbose=False) for f in tqdm(full_paths)]
    for idx, fn in enumerate(all_result_files):
        assert fn[:15] == "checkpoint_step"
        step_num = int(fn[15:-4])
        Stats[step_num] = {"file_path": os.path.join(result_path, fn)}
        Stats[step_num]["result"] = results[idx]

    # return results
    Stats = dict(sorted(Stats.items()))
    ResultSummary = {}
    stat_names = ["ppl", "all_iou", "any_iou", "g_score_all", "g_score_any", "bridge_raw_top1_norm_prob", "bridge_raw_top1_kl", "bridge_raw_top1_any_iou", "bridge_raw_top1_any_box_hit","bridge_proposed_norm_prob", "bridge_proposed_kl", "bridge_proposed_any_iou", "mask_hit", "all_box_hit", "all_hit", "any_box_hit", "any_hit", "soft_match_l1", "soft_match_l2", "soft_match_kl", "all_hit_top3", "all_hit_top5", "all_hit_top10", "any_hit_top3", "any_hit_top5", "any_hit_top10", "word_dis", "amb_any_iou", "amb_any_box_hit", "mask_hit_top3", "mask_hit_top5", "mask_hit_top10"]
    for test_idx in range(len(slicings)):
        ResultSummary[slicings[test_idx].name] = {name: [] for name in stat_names}
        ResultSummary[slicings[test_idx].name]['step'] = []
        for k, v in Stats.items():
            ResultSummary[slicings[test_idx].name]["step"].append(k)
            for stat_name in stat_names:
                ResultSummary[slicings[test_idx].name][stat_name].append(v['result'][test_idx][stat_name])
    torch.save(ResultSummary, "ResultSummary_NoGrounding_Finetuned.pth")
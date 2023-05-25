#!/bin/bash
#SBATCH --job-name=Cont-Eval
#SBATCH --partition=spgpu
#SBATCH --account=chaijy2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-62
#SBATCH --output="/scratch/chaijy_root/chaijy2/jiayipan/ACL/logs/slurm-%A_%a.out"
module restore teach


cd "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/RefCloze/"
source "/sw/pkgs/arc/python3.9-anaconda/2021.11/bin/activate" na
python aoa_evaluate.py --model_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/per_class/checkpoint/pizza.pth" --log_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/pizza.pth" --dataset_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/refcloze_data/Archive/flickr_unseen_pizza_dataset.json"
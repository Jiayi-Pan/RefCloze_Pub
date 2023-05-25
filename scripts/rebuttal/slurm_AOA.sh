#!/bin/bash
#SBATCH --job-name=AOA-Eval-unseen
#SBATCH --partition=spgpu
#SBATCH --account=chaijy2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-162
#SBATCH --output="/scratch/chaijy_root/chaijy2/jiayipan/ACL/logs/slurm-%A_%a.out"
module restore teach


CKPT_DIR="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/outputs/Flickr-gl/"
CKPT_FILES=( $( ls $CKPT_DIR ) )
THIS_FILE="${CKPT_FILES[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_ROOT="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/AOA/flickr_gl/"
SEEN_LOG_DIR="${OUTPUT_ROOT}seen/"
UNSEEN_LOG_DIR="${OUTPUT_ROOT}unseen/"

echo "==="
echo "checkpoint file name:"
echo $THIS_FILE
echo "==="
echo "result path:"
echo $SEEN_LOG_DIR
echo $UNSEEN_LOG_DIR
# LOG_FILE="$LOG_DIR${CKPT_FILES[SLURM_ARRAY_TASK_ID]}"

cd "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/RefCloze/"
source "/sw/pkgs/arc/python3.9-anaconda/2021.11/bin/activate" na


python aoa_evaluate.py --model_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/origin/checkpoint/15w.pth" --log_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/rebuttal_acl_beautiful/standard.pth" --dataset_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/refcloze_data/flickr_all_eval_beautiful_only_refcloze_dataset_.json"

python aoa_evaluate.py --model_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/origin/checkpoint/15w_gl_probed.pth" --log_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/rebuttal_acl_beautiful/gl.pth" --dataset_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/refcloze_data/flickr_all_eval_beautiful_only_refcloze_dataset_.json"
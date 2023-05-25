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
#SBATCH --array=0-1
#SBATCH --output="/scratch/chaijy_root/chaijy2/jiayipan/ACL/logs/slurm-%A_%a.out"
module restore teach


# SPLIT_Root="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/origin/"
# SPLIT_Root="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/all/"
SPLIT_Root="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/Last_Exps/"
# SPLIT_Root="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/per_class/"
# SPLIT_Root="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/stream/"
CKPT_FILES=( $( ls $SPLIT_Root ) )
CKPT_DIR="${SPLIT_Root}checkpoint/"
CKPT_FILES=( $( ls $CKPT_DIR ) )
THIS_FILE="${CKPT_FILES[$SLURM_ARRAY_TASK_ID]}"
SEEN_LOG_DIR="${SPLIT_Root}seen/"
UNSEEN_LOG_DIR="${SPLIT_Root}unseen/"

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
python aoa_evaluate.py --model_path "${CKPT_DIR}${THIS_FILE}" --log_path "${UNSEEN_LOG_DIR}${THIS_FILE}" --dataset_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/refcloze_data/0120_Unseen_All.json"
python aoa_evaluate.py --model_path "${CKPT_DIR}${THIS_FILE}" --log_path "${SEEN_LOG_DIR}${THIS_FILE}" --dataset_path "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/refcloze_data/flickr_seen_refcloze_dataset.json"
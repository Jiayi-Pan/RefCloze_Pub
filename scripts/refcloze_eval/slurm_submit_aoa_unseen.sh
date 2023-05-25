#!/bin/bash
# submit_array.sh

#SBATCH --job-name=AOA-Eval
#SBATCH --partition=spgpu
#SBATCH --account=chaijy2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output="/scratch/chaijy_root/chaijy2/jiayipan/ACL/slurm-%A_%a.out"
module restore teach

# AOA_CKPT_DIR='/scratch/chaijy_root/chaijy2/jiayipan/outputs/0710_flickr_only/AOAs/AOA_Remake_Checkpoints/'
# LOG_DIR="/scratch/chaijy_root/chaijy2/jiayipan/outputs/0710_flickr_only/AOAs/AOA_for_COCO_Ana_Dataset/AOA_Results_with_Box/"
# DATASET_DIR = "/nfs/turbo/coe-chaijy/jiayipan/Vision-Language-Project/EMNLP/Evaluation/dataset/Ana_RefCOCO_Dataset.json"

# AOA_CKPT_DIR="/scratch/chaijy_root/chaijy2/jiayipan/outputs/model-mlm32-flickr-scratch-only-wo-vl-mapping/AOAs/Checkpoints/"
# LOG_DIR="/scratch/chaijy_root/chaijy2/jiayipan/outputs/model-mlm32-flickr-scratch-only-wo-vl-mapping/AOAs/Results/"
# AOA_CKPT_DIR="/scratch/chaijy_root/chaijy2/jiayipan/outputs/prob/Checkpoints"

AOA_CKPT_DIR="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/ablation_freeze/checkpoints/"
LOG_DIR="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/ablation_freeze/unseen/"
DATASET_DIR="/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/refcloze_data/flickr_unseen_pizza_dataset.json"

CKPT_FILES=( $( ls $AOA_CKPT_DIR ) )
THIS_FILE="${CKPT_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "==="
echo "checkpoint file name:"
echo $THIS_FILE
THIS_CKPT_FILE="$AOA_CKPT_DIR$THIS_FILE"
echo "==="
echo "checkpoint file path:"
echo $THIS_CKPT_FILE
LOG_FILE="$LOG_DIR$THIS_FILE"
echo "==="
echo "result path:"
echo $LOG_FILE
# LOG_FILE="$LOG_DIR${CKPT_FILES[SLURM_ARRAY_TASK_ID]}"

# cd "/nfs/turbo/coe-chaijy/jiayipan/Vision-Language-Project/EMNLP/Evaluation/VL-2-L-Uni"
# python Ultimate_Evaluator_with_KL_softeval.py --model_path "$THIS_CKPT_FILE" --log_path "$LOG_FILE"
cd "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/RefCloze/"
source "/sw/pkgs/arc/python3.9-anaconda/2021.11/bin/activate" na
python aoa_evaluate.py --model_path "$THIS_CKPT_FILE" --log_path "$LOG_FILE" --dataset_path "$DATASET_DIR"
sleep 10
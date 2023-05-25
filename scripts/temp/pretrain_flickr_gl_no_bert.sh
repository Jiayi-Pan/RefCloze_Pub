# this just trains for the first 500 steps because the checkpoints aren't saved!
TORCHELASTIC_ERROR_FILE=./temp_file
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=23333 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=10.164.10.19 \
main.py --dataset_config configs/pretrain_test.json \
--run_name MaskDETR_ACL_BL \
--no_vl_mapping_stage1 \
--save_for_aoa \
--wandb \
--lr 0.0001 \
--text_encoder_lr 0.00001 \
--clip_max_norm 5 \
--num_queries 50 \
--backbone resnet50 \
--text_encoder_type roberta-base \
--freeze_vision_encoder \
--batch_size 16 \
--mlm_loss \
--mlm_loss_coef 32 \
--mlm_other_prob 0.10 \
--mlm_conc_noun_prob 0.4 \
--vl_enc_layers 4 \
--obj_dec_layers 4 \
--lang_dec_layers 4 \
--dim_feedforward 2048 \
--hidden_dim 512 \
--num_workers 4 \
--pin_mem \
--epochs 200 \
--amp \
--shuffle \
--text_from_scratch \
--output-dir /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/outputs/BertLess
torchrun \
--nnodes=1 \
--nproc_per_node=8 \
--standalone \
main.py --dataset_config configs/pretrain_test.json \
--run_name MaskDETR_sweep_param \
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
--num_workers 4 \
--pin_mem \
--amp \
--shuffle \
--output-dir /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/outputs/sweep3 \
${@:1}

# --rdzv_endpoint="10.164.10.14" \
# --rdzv_backend=c10d \
# --rdzv_id=23333 \
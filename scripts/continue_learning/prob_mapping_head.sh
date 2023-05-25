for file in /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/outputs/Prob/Checkpoint/*; do
    echo "${file}"
    TORCHELASTIC_ERROR_FILE=./temp_file
    torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --standalone \
    main.py --dataset_config configs/pretrain_test.json \
    --run_name "RefCloze Probe" \
    --wandb \
    --save_for_aoa \
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
    --epochs 1 \
    --amp \
    --shuffle \
    --output-dir "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/outputs/Prob/out/" \
    --load "${file}" \
    --no_vl_mapping_stage2

    sleep 30
done
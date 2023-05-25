word_list=("brush" "teacher")

for word in "${word_list[@]}"
do
    echo $word
    TORCHELASTIC_ERROR_FILE=./temp_file
    torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --standalone \
    main.py --dataset_config "continue_configs/${word}.json" \
    --run_name MaskDETR_ACL_Per_Class \
    --is_gl \
    --learn_word_till_converge \
    --no_dense_sup \
    --wandb \
    --lr 0.0001 \
    --text_encoder_lr 0.00001 \
    --clip_max_norm 5 \
    --num_queries 50 \
    --backbone resnet50 \
    --text_encoder_type roberta-base \
    --freeze_vision_encoder \
    --batch_size 8 \
    --mlm_loss \
    --mlm_loss_coef 32 \
    --mlm_other_prob 0.10 \
    --mlm_conc_noun_prob 0.4 \
    --vl_enc_layers 4 \
    --obj_dec_layers 4 \
    --lang_dec_layers 4 \
    --dim_feedforward 2048 \
    --hidden_dim 512 \
    --num_workers 2 \
    --pin_mem \
    --epochs 1000 \
    --amp \
    --load /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/outputs/final_models/Checkpoints/15w_gl_probed.pth \
    --output-dir /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/per_class/checkpoint 
    # --output-dir /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/outputs/per_class/
done
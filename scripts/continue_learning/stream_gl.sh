unseen_word_list=('pizza' 'crosswalk' 'cheese' 'star' 'classroom' 'fluffy' 'circular' 'sink' 'doctor' 'foreign' 'diverse' 'product' 'stove' 'steep' 'warm' 'donkey' 'aged' 'barefoot' 'elephant' 'heart' 'gym' 'security' 'cafe' 'teacher' 'button' 'student' 'newborn' 'pan' 'telephone' 'bamboo' 'brush')

for i in $(seq 0 $((${#unseen_word_list[@]}-2))); do
  echo ${unseen_word_list[i]} ${unseen_word_list[i+1]}
    TORCHELASTIC_ERROR_FILE=./temp_file
    torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --standalone \
    main.py --dataset_config "continue_configs/${unseen_word_list[i+1]}.json" \
    --run_name MaskDETR_ACL_Stream \
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
    --load "/scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/stream/checkpoint/${unseen_word_list[i]}_gl.pth" \
    --is_gl \
    --output-dir /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/continue/stream/checkpoint
done
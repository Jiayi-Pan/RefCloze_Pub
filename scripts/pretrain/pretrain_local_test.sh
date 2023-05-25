TORCHELASTIC_ERROR_FILE=./error_file
torchrun main.py --dataset_config configs/pretrain_test.json \
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
--epochs 2 \
--amp \
--shuffle \
--run_name flickr_local_test \
--output-dir /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/outputs/test
--save_for_aoa \
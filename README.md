# W2 Bert

``` bash
pip install wandb prettytable spacy pycocotools einops timm scipy tqdm
```
## Note
- /scratch/chaijy_root/chaijy2/jiayipan/ACL/RefCloze/pretrain_data/preprocessed_annotations/final_flickr_separate_GT_train.json is modified from origin
    - basicially left out < 5% to study continual learning


## ChangeLog 

## Terminology
### MDETR
- mask: a binary mask of shape [batch_size x H x W], containing True on padded pixels, 

### Sweep
After some [simple hyper-parameter search](https://wandb.ai/jiayipan/RefCloze/sweeps/tud7cuf5?workspace=user-jiayipan), we found the following hyper-parameters perform the best:
- lr: 1e-4
- text_encoder_lr: 1e-5
- clip_max_norm: 5 
- num_queries: 50

<!-- TODO check if reducing dim is working -->
--no_vl_mapping_stage_1

### Data

- continue
    - SampledNumX (X = 16, 24, 32) where is 8?
        - per_class: where we feed in a new class
        - stream: where we feed in a stream of data in new class
    - all: deprecated, where we feed in all data without continue streaming
    - origin
        - seen/15w.pth / seen/15w_gl_probed.pth: performance for 15w steps model
        - unseen/15w.pth / unseen/15w_gl_probed.pth performance for 15w steps model
- AOA

### AOA
Scripts to run AOA are in RefCloze/analysis_utils
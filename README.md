# W2 BERT: World to Words: Language Modeling Down to the Ground
[Website] • [Colab] • [Huggingface] • [Paper]

[Ziqiao Ma](https://mars-tin.github.io/)\*, [Jiayi Pan](https://www.jiayipan.me/)\*, [Joyce Chai](https://web.eecs.umich.edu/~chaijy/) (\* equal contribution)

**SLED Lab @ University of Michigan**


## Online Demo

## Introduction

## Getting Started
The easiest way to play with our model is through HuggingFace Space or Colab. 

However, if you are interested in reproducing our project, please follow the instructions below, which will guide you through the installation, pre-training, inference and evaluation stages.

### Installation
Clone our repository, create a python environment and install and install the required packages.

```
# After installing pytorch 
pip install requirements.txt
```


### Pre-training
We provide model weights for our pre-trained model. You can download the weights from [here](https://drive.google.com/drive/folders/1-0Z3Z3Q3Z3Q3Z3Q3Z3Q3Z3Q3Z3Q3Z3Q3?usp=sharing) and put them under the `pretrain_data` folder.

If you are interested in training the model yourself, please follow the instructions in [Pre-training Instructions](scripts/pretrain/README.md). 


### Inference

### Evaluation

#### Evaluation Dataset

#### Evaluation Script



## Citation
If you find our project useful, please give it a star and cite as follows :)

```bibtex
CITATION TO ADD
```

## Acknowledgement
Our project is built upon [MDETR](https://github.com/ashkamath/mdetr) and [DETR](https://github.com/facebookresearch/detr). We thank the authors for their great work!

## License
The project is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
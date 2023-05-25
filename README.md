# World to Words: Language Modeling Down to the Ground

TODO: 
1. Finish the sections
2. Rename the model/dataset/project

How much do we want to coverage:
- Training process
- How about eval datasets
- How about AOA/ablation/other analysis?

[Website]() • [Model Demo]() • [Dataset Demo]() • [Paper]()

[Ziqiao Ma](https://mars-tin.github.io/)\*, [Jiayi Pan](https://www.jiayipan.me/)\*, [Joyce Chai](https://web.eecs.umich.edu/~chaijy/) (\* denotes equal contribution)

**SLED Lab @ University of Michigan**

![Model](docs/images/model.png)

## Online Demo

### RefCloze Dataset

Our RefCloze dataset is available on [HuggingFace Dataset](https://huggingface.co/datasets/zma/refcloze) and can be visualized through [HuggingFace Space](https://huggingface.co/spaces/zma/refcloze).

### W2-Bert

You can play with our model through [HuggingFace Space]() or [Colab]().

## Introduction

## Getting Started

The easiest way to play with our model/dataset is through HuggingFace Space or Colab.

However, if you are interested in reproducing our project, please follow the instructions below, which will guide you through the installation, pre-training, inference and evaluation stages.

### Installation

Clone our repository and create the python environment, you can install the required packages by

```bash
pip install requirements.txt
```

### Pre-training

We provide model weights for our pre-trained model. You can download the weights from [here](https://drive.google.com/drive/folders/1-0Z3Z3Q3Z3Q3Z3Q3Z3Q3Z3Q3Z3Q3Z3Q3?usp=sharing) and put them under the `pretrain_data` folder.

If you are interested in training the model yourself, please follow the instructions in [Pre-training Instructions](scripts/pretrain/README.md).

### Inference

### Evaluation Dataset

### Model Evaluation

## Citation

If you find our project useful, please give us a star and cite as follows :)

```bibtex
CITATION TO BE ADDED
```

## Acknowledgement

Our project is built upon [MDETR](https://github.com/ashkamath/mdetr), [DETR](https://github.com/facebookresearch/detr) and many others. We thank the authors for their great work!

## License

The project is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

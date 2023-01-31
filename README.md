This repository releases the code of our paper ["Image Shortcut Squeezing:
Countering Perturbative Availability Poisons with Compression"]().


### Overview

This implementation verifies that 12 state-of-the-art Perturvative Availability Poisoning (PAP) 
methods are vulnerable to Image Shortcut Squeezing (ISS), which is based on simple compression (i.e., grayscale compression, JPEG compression, and bit depth reduction). For example, 
on average, ISS restores the CIFAR-10 model accuracy to 81.73%, surpassing the previous best preprocessing-based countermeasures by 37.97% absolute.
We hope that further studies could consider various (simple) countermeasures during the development of new poisoning methods.

###  Categorization of existing poisoning methods
We carry out a systematic analysis of compression-based countermeasures for PAP, including data augmentations and adversarial training. 
We identify the strong dependency of the perturbation frequency patterns on the surrogate model property: perturbations that are generated on slightly-trained surrogates exhibit spatially low-frequency patterns, and Poisons that are generated on fully-trained surrogates, and perturbations
trained on fully-trained surrogates exhibit spatially high-frequency patterns, as shown in the Figure below. 

<img src="/images/examples.png" alt="examples">


### Evaluation on 12 poisoning methods by ISS.

| Poisons \ Countermeasures  | w/o  | Grayscale | JPEG-10|
| ------------- | ------------- |------------- |-------------|
| Clean (no poison)  | 94.68  | 92.41 | 85.38 |
| [Deep Confuse](https://papers.nips.cc/paper/2019/hash/1ce83e5d4135b07c0b82afffbe2b3436-Abstract.html)  $(L_{\infty} = 8)$ | 16.30  |93.07| 81.84|
| [NTGA](https://proceedings.mlr.press/v139/yuan21b)  $(L_{\infty} = 8)$ | 42.46  | 74.32 | 69.49|
| [EM](https://openreview.net/forum?id=iAmZUo0DxC0)  $(L_{\infty} = 8)$ | 21.05  |93.01|81.50|
| [REM](https://openreview.net/forum?id=baUQQPwQiAg)  $(L_{\infty} = 8)$ | 25.44  |92.84|81.50|
| [ShortcutGen](https://arxiv.org/abs/2211.01086)  $(L_{\infty} = 8)$ | 33.05  |86.42|79.49|
| [TensorClog](https://ieeexplore.ieee.org/document/8668758) $(L_{\infty} = 8)$  | 88.70| 79.75|85.29|
| [Hypocritical](https://arxiv.org/abs/2102.04716)  $(L_{\infty} = 8)$ | 71.54|61.86|85.45|
| [TAP](https://proceedings.neurips.cc/paper/2021/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html)  $(L_{\infty} = 8)$ | 8.17|9.11|83.87|
| [SEP](https://openreview.net/forum?id=9MO7bjoAfIA)  $(L_{\infty} = 8)$ | 3.85 |3.57|84.37|
| [LSP](https://dl.acm.org/doi/10.1145/3534678.3539241)  $(L_{2} = 1)$ | 19.07 |82.47|83.01|
| [AR](https://openreview.net/forum?id=1vusesyN7E)  $(L_{2} = 1)$ | 13.28  |34.04|85.15|
| [OPS](https://openreview.net/forum?id=p7G8t5FVn2h)  $(L_{0} = 1)$ | 36.55  |42.44|82.53|


### How to apply ISS on poisons?

Prepare poisoned images as `.png` files in folder `PATH/TO/POISON_FOLDER` following the order of original CIFAR-10 dataset.

To train on grayscaled poisons: 

`python main.py --exp_type $TYPEOFPOISONS --poison_path PATH/TO/POISON_FOLDER --poison_rate 1 --net resnet18 --grayscale True --exp_path PATH/TO/SAVE/RESULTS/`


To train on JPEG compressed poisons: 

`python main.py --exp_type $TYPEOFPOISONS --poison_path PATH/TO/POISON_FOLDER --poison_rate 1 --net resnet18 --jpeg 10 --exp_path PATH/TO/SAVE/RESULTS/`
   
   
To train on bit depth reducted poisons: 

`python main.py --exp_type $TYPEOFPOISONS --poison_path PATH/TO/POISON_FOLDER --poison_rate 1 --net resnet18 --BDR 2 --exp_path PATH/TO/SAVE/RESULTS/`


### An example:

We provide an example of training on CIFAR-10 poisoned by [Targeted Adversarial Poisoning (TAP)](https://proceedings.neurips.cc/paper/2021/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html).
Poisons are taken directlly from the official [TAP GitHub repository](https://github.com/lhfowl/adversarial_poisons). After `bash run.sh`, results can be found in `experiments/TAP/`

### Cite our work:

Please cite our paper if you use this implementation in your research.

```
@misc{liu2023ISS,
      title={Image Shortcut Squeezing: Countering Perturbative Availability Poisons with Compression}, 
      author={Zhuoran Liu and Zhengyu Zhao and Martha Larson},
      eprint={},
      archivePrefix={arXiv},
      year={2023}
}
```


### Acknowledgement:
Training code adapted from kuangliu's repository [Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar)

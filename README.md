# HyperMixer: An MLP-based Green AI Alternative to Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ChristophReich1996/HyperMixer/blob/master/LICENSE)

This is a very quick (and maybe wrong) PyTorch reimplementation of the [HyperMixer](https://arxiv.org/pdf/2203.03691.pdf).

# Usage

```python
import torch
from hyper_mixer import HyperMixerBlock

hyper_mixer_block = HyperMixerBlock(dim=32)
input = torch.rand(3, 256, 32)
output = hyper_mixer_block(input)
print(output.shape)
```

# Reference

```bibtex
@article{Liu2021,
    title={{HyperMixer: An MLP-based Green AI Alternative to Transformers}},
    author={Mai, Florian and Pannatier, Arnaud and Fehr, Fabio and Chen, Haolin and Marelli, Francois 
    and Fleuret, Francois and Henderson, James},
    journal={arXiv preprint arXiv:2203.03691},
    year={2022}
}
```

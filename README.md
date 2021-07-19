# Pyg-TextGCN 

An *efficient* and *simplify* re-implement TextGCN with Pytorch-geometric. Fork from [PyTorch_TextGCN](https://github.com/chengsen/PyTorch_TextGCN).
<!-- This one is more faster than others(include raw implement) in the own limited testing. -->

## Requirements

This project was built with:

- Python 3.7.0
- Pytorch 1.9.0
- scikit-learn 0.24.1
- torch-geometric 1.7.2
- numpy 1.19.5
- pandas 1.1.5 

## Quick Start

Process the data first, `python data_processor.py` (Already done)

Generate graph, `python build_graph.py` (Already done)

Training model, `python run.py`

## References

[Yao et al.: Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)

[Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)

[PyTorch_TextGCN](https://github.com/chengsen/PyTorch_TextGCN)

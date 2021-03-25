# ANRL-PyTorch
An implementation of ANRL based on PyTorch. Most code of ANRL model in this repository comes from [cszhangzhen/ANRL](https://github.com/cszhangzhen/ANRL).

Thanks to [Stonesjtu/Pytorch-NCE](https://github.com/Stonesjtu/Pytorch-NCE), We can easily achieve our aim similar to `tf.nn.sampled_softmax_loss`.

Original paper [ANRL: Attributed Network Representation Learning via Deep Neural Networks](https://www.ijcai.org/Proceedings/2018/0438.pdf)

## Usage
`pip install -r requirements.txt`
`python main.py`

## Attention
1. We did not use exactly the same model weight initialization method as in ANRL.
2. The underlying implementation of PyTorch may result in different results from tensorflow under the same hyper-parameter settings。

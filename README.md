# moonshine

Code used to produce https://arxiv.org/abs/1711.02613

## Installation Instructions

Best done with conda. Make sure your conda is up to date.

Make a new environment then activate it. Python version probably doesn't matter but I use 2 for no apparent reason.
```
conda create -n torch python=2
source activate torch
```
then

```
conda install pytorch torchvision -c pytorch
pip install tqdm
pip install tensorboardX
pip install tensorflow
```

## Training a Teacher

In general, the following code trains a teacher network:

```
python main.py <DATASET> teacher --conv <CONV-TYPE> -t <TEACHER_CHECKPOINT> --wrn_depth <TEACHER_DEPTH> --wrn_width <TEACHER_WIDTH>
```

In the paper, results are typically reported using a standard 40-2 WRN,
which would be the following (on cifar-10):

```
python main.py cifar10 teacher --conv Conv -t wrn_40_2 --wrn_depth 40 --wrn_width 2
```

## Training a Student

Students can be trained using KD (by setting alpha>0) and/or AT (by setting beta>0) as:

```
python main.py <DATASET> student --conv <CONV-TYPE> -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --wrn_depth <STUDENT_DEPTH> --wrn_width <STUDENT_WIDTH> --alpha <ALPHA for KD> --beta <BETA for AT>
```
  
Note: the AT method uses KD by default, so to turn it off, set alpha to 0

As an example, this would train a model with the same structure as the
teacher network, but using a bottleneck grouped + pointwise convolution as
a substitute for the full convolutions in the full network with attention transfer:

```
python main.py cifar10 student --conv G8B2 -t wrn_40_2 -s wrn_40_2.g8b2.student --wrn_depth 40 --wrn_width 2 --alpha 0. --beta 1e3
```

## Acknowledgements

Code has been liberally borrowed from other repos.

A non-exhaustive list follows:

```
https://github.com/szagoruyko/attention-transfer
https://github.com/kuangliu/pytorch-cifar
https://github.com/xternalz/WideResNet-pytorch
```

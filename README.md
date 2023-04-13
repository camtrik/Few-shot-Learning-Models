# Meta-Learning Models for Few-shot Classification
This repository contains several meta-learning models used for few-shot tasks. Also includes pretrain process i.e. training in a classical way and test on 5-way 5-shot task directly. 

Combine the pretraining and meta-training would make the performance better.

## Running
### Environment
- python 3.9.3
- pytorch 1.14.0
- wandb

### Datasets
- miniImageNet (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
- tieredImageNet (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))

### Training
**Pretrain**
```
python pretrain.py --config config/pretrain_mini.yaml
```
**Meta Training**
```
python train.py --config/train_proto_mini.yaml
```

### Testing
The trained model would be saved in logs/.../... .pth
Modify the load path in the test.yaml to load the model
```
python test.py --config/test.yaml
```

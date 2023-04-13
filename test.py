import yaml 
import os 
import shutil
import argparse

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import models
import datasets 
from utils.functions import *
from utils.split import *
from datasets.few_shot_sampler import FewShotSampler


def main(config):
    # few-shot settings 
    n_batch = 50
    n_way = 5
    n_support = args.shot
    n_query = 15
    n_episodes = 2
    test_epochs = 5

    # datasets and dataloaders
    dataset = datasets.make_dataset(config['dataset'], **config['dataset_args'])
    sampler = FewShotSampler(dataset.label, n_batch, n_way, n_support, n_query, n_episodes)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)
    log('test dataset: {}, number: {}, classes: {} '.format(dataset[0][0].shape, len(dataset), dataset.n_classes))

    # model 
    model = models.make_model(config['model'], **config['model_args'])
    if config.get('load'):
        print("Loading model from {}".format(config['load']))
        model_save = torch.load(config['load'])
        model.load_state_dict(model_save)

    model.eval()
    log('model: {}, params: {}'.format(config['model'], compute_params(model.encoder)))

    log_keys = ['test_loss', 'test_acc']
    test_log = {key: [] for key in log_keys}
    

    for epoch in range(test_epochs):
        aves = {k: Averager() for k in log_keys}
        for data, _ in tqdm(loader, leave=False):
            x_support, x_query = split_support_query(data.cuda(), n_way, n_support, n_query, n_episodes)
            labels = make_label(n_way, n_query, n_episodes).cuda()
            with torch.no_grad():
                logits = model(x_support, x_query).view(-1, n_way)
                loss = F.cross_entropy(logits, labels)
                acc = compute_acc(logits, labels)

            aves['test_loss'].add(loss.item())
            aves['test_acc'].add(acc)
        

        for key, value in aves.items():
            aves[key] = value.item()
            test_log[key].append(aves[key])

        print('epoch: {}, test_loss: {:.4f}, test_acc: {:.4f}'.format(epoch, aves['test_loss'], aves['test_acc']))
    # compute the mean of test_acc and test_loss in test_log

    test_acc = torch.mean(torch.tensor(test_log['test_acc']))
    test_loss = torch.mean(torch.tensor(test_log['test_loss']))
    
    print('test_acc: {:.4f}, test_loss: {:.4f}'.format(test_acc, test_loss))

if __name__ == '__main__':
    config = yaml.load(open('config/test.yaml'), Loader=yaml.FullLoader)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=5)
    args = parser.parse_args()

    main(config)
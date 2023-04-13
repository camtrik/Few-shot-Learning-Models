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

import wandb 

def main(config):
    save_path = os.path.join('logs', config['model'] + '_' + config['train'] + '_' + str(config['n_way']) + 'way_' + str(config['n_support']) + 'shot')
    is_path(save_path)
    set_log_path(save_path)
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    
    # wandb settings
    wandb.init(project='few-shot', name=config['model'] + '_' + config['train'] + 
               '_' + str(config['n_way']) + 'way_' + str(config['n_support']) + 'shot')
    wandb.config.update(config)
    
    

    # few-shot settings
    n_batch = config['n_batch']
    n_way = config['n_way']
    n_support = config['n_support']
    n_query = config['n_query']
    n_episodes = config['n_episodes']

    # datasets and dataloaders
    train_dataset = datasets.make_dataset(config['train'], **config['train_args'])
    train_sampler = FewShotSampler(train_dataset.label, n_batch, n_way, n_support, n_query, n_episodes)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    log('train dataset: {}, number: {}, classes: {} '.format(train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))

    val_dataset = datasets.make_dataset(config['val'], **config['val_args'])
    val_sampler = FewShotSampler(val_dataset.label, n_batch, n_way, n_support, n_query, n_episodes)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    log('val dataset: {}, number: {}, classes: {} '.format(val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))

    
    
    # model
    model = models.make_model(config['model'], **config['model_args'])
    if config.get('load'):
        print("Loading model from {}".format(config['load']))
        model_save = torch.load(config['load'])
        model.load_state_dict(model_save)

    log('model: {}, params: {}'.format(config['model'], compute_params(model.encoder)))

    # other settings
    optimizer, lr_scheduler = make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.0

    log_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    train_log = {key: [] for key in log_keys}
    
    for epoch in range(max_epoch):
        # create a ave to store the train loss and train acc, and the val loss and val acc
        aves = {key: Averager() for key in log_keys}
        model.train()
        for data, _ in tqdm(train_loader, desc='train', leave=False):
            x_support, x_query = split_support_query(data.cuda(), n_way, n_support, n_query, n_episodes)
            labels = make_label(n_way, n_query, n_episodes).cuda()

            # similarity shape to (n_query * n_way * n_episodes, n_way)
            logits = model(x_support, x_query).view(-1, n_way)
            loss = F.cross_entropy(logits, labels)
            acc = compute_acc(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['train_loss'].add(loss.item())
            aves['train_acc'].add(acc)

        model.eval()
        
        for data, _ in tqdm(val_loader, desc='val', leave=False):
            x_support, x_query = split_support_query(data.cuda(), n_way, n_support, n_query, n_episodes)
            labels = make_label(n_way, n_query, n_episodes).cuda()

            with torch.no_grad():
                logits = model(x_support, x_query).view(-1, n_way)
                loss = F.cross_entropy(logits, labels)
                # print(logits.shape, labels.shape)
                acc = compute_acc(logits, labels)

            aves['val_loss'].add(loss.item())
            aves['val_acc'].add(acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        for key, value in aves.items():
            aves[key] = value.item()
            train_log[key].append(aves[key])

        log('epoch: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}'.format(
            epoch, aves['train_loss'], aves['train_acc'], aves['val_loss'], aves['val_acc']))
    
        # wandb log 
        wandb.log({'train_loss': aves['train_loss'], 'train_acc': aves['train_acc'],
                     'val_loss': aves['val_loss'], 'val_acc': aves['val_acc']})

        if aves['val_acc'] > max_va:
            max_va = aves['val_acc']
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            log('save best model! ')

if __name__ == "__main__":
    # config = yaml.load(open('config/train_proto.yaml'), Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train_proto_mini.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    main(config)
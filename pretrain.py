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
    save_path = os.path.join('logs', 'pretrain_' + config['model'] + '_' + config['train'])
    is_path(save_path)
    set_log_path(save_path)
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    
    n_batch = 200
    n_way = 5
    n_support = 5
    n_query = 15
    n_episodes = 2


    # wandb settings
    wandb.init(project='pretrain-few-shot', name=config['model'] + '_' + config['train'])
    wandb.config.update(config)

    # Datasets
    train_dataset = datasets.make_dataset(config['train'], **config['train_args'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    log('train dataset: {}, number: {}, classes: {} '.format(train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))

    eval_val = False
    if config.get('val'):
        eval_val = True
        val_dataset = datasets.make_dataset(config['val'], **config['val_args'])
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        log('val dataset: {}, number: {}, classes: {} '.format(val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))

    # few-shot eval 
    eval_fs = False
    if config.get('fs'):
        eval_fs = True
        fs_dataset = datasets.make_dataset(config['fs'], **config['fs_args'])
        fs_sampler = FewShotSampler(fs_dataset.label, n_batch, n_way, n_support, n_query, n_episodes)
        fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler, num_workers=8, pin_memory=True)
        log('fs dataset: {}, number: {}, classes: {} '.format(fs_dataset[0][0].shape, len(fs_dataset), fs_dataset.n_classes))

    # Model
    model = models.make_model(config['model'], **config['model_args'])
    if config.get('load'):
        print("Loading model from {}".format(config['load']))
        model_save = torch.load(config['load'])
        model.load_state_dict(model_save)

    if eval_fs:
        ef_epoch = config['eval_fs_epoch']
        fs_model = models.make_model('protonet', encoder=None)
        fs_model.encoder = model

        # log dataset
        log('fs dataset: {}, number: {}, classes: {} '.format(fs_dataset[0][0].shape, len(fs_dataset), fs_dataset.n_classes))


    log('model: {}, params: {}'.format(config['model'], compute_params(model.encoder)))

    # other settings
    optimizer, lr_scheduler = make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])

    max_epoch = config['max_epoch']
    max_va = 0

    for epoch in range(1 + max_epoch):
        
        aves_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc', '5-shot acc']
        aves = {k: Averager() for k in aves_keys}
        
        model.train()
        for data, label in tqdm(train_loader, desc='train', leave=False):
            data, label = data.cuda(), label.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            aves['train_loss'].add(loss.item())
            aves['train_acc'].add(acc)

        if eval_val:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = compute_acc(logits, label)

                aves['val_loss'].add(loss.item())
                aves['val_acc'].add(acc)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch):
            fs_model.eval()
            for data, _ in tqdm(fs_loader, desc='fs', leave=False):
                x_shot, x_query = split_support_query(data.cuda(), n_way, n_support, n_query, n_episodes)
                labels = make_label(n_way, n_query, n_episodes).cuda()
                with torch.no_grad():
                    logits = fs_model(x_shot, x_query).view(-1, n_way)
                    # print(logits.shape, labels.shape)
                    acc = compute_acc(logits, labels)
                
                aves['5-shot acc'].add(acc)

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        for key, value in aves.items():
            if epoch % ef_epoch == 0 or epoch == max_epoch:
                aves[key] = value.item()
            else:
                if key != '5-shot acc':
                    aves[key] = value.item()

        log('epoch: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}'.format(epoch,
             aves['train_loss'], aves['train_acc'], aves['val_loss'], aves['val_acc']))
        
        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch):
            log('epoch: {}, 5-shot acc: {}'.format(epoch, aves['5-shot acc']))
        # wandb log 
        wandb.log({'train_loss': aves['train_loss'], 'train_acc': aves['train_acc'],
                    'val_loss': aves['val_loss'], 'val_acc': aves['val_acc']})
        
        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch):
            wandb.log({'5-shot acc': aves['5-shot acc']})

        if aves['val_acc'] > max_va:
            max_va = aves['val_acc']
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            log('save best model! ')

        if epoch == max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, 'last_model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/pretrain_mini.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from utils.model import PushNet
from utils.dataloader import PushNetDataset
import multiprocessing
import yaml

def test_loop(test_loader, model, loss_fn):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    test_loss, test_acc, test_ap = 0, 0 ,0
    true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
    model.train(False)
    pbar = tqdm(test_loader)    
    pbar.set_description('Valid Error: | Loss: {:.4f} | Acc: {:.4f} | Prec: {:.4f} | Recall: {:.4f} | AP: {:.4f}'.format(test_loss, test_acc, 0, 0, test_ap))
    with torch.no_grad():
        for images, velocities, labels in pbar:
            images = images.to(DEVICE)
            velocities = velocities.to(DEVICE)
            labels = labels.to(DEVICE)
            # Forward pass
            pred = model(images, velocities)
            loss = loss_fn(pred, labels)

            # Accumulate loss 
            test_loss += loss.item()

            # Accumulate accuracy
            labels = torch.argmax(labels, dim=1)
            results = torch.softmax(pred, dim=1)
            results = torch.stack([results[:,1], labels], dim=-1)
            results = results[torch.sort(results[:,0], descending=True)[1]]
            results = torch.where(results < 0.5, 0, 1)
            true_positives = torch.sum(torch.logical_and(results[:,0] == 1, results[:,1] == 1)).item()
            false_positives = torch.sum(torch.logical_and(results[:,0] == 1, results[:,1] == 0)).item()
            false_negatives = torch.sum(torch.logical_and(results[:,0] == 0, results[:,1] == 1)).item()
            true_negatives = torch.sum(torch.logical_and(results[:,0] == 0, results[:,1] == 0)).item()

            recall_num = torch.tensor([true_positives + false_negatives]).repeat_interleave(results.shape[0]).to(DEVICE)
            prec_num = torch.zeros(results.shape[0]).to(DEVICE)
            prec_num[torch.where(results[:,0] == 1)] = 1
            prec_num = torch.cumsum(prec_num, dim=0)
        
            TP = torch.zeros(results.shape[0]).to(DEVICE)
            TP[torch.logical_and(results[:,0] == 1, results[:,1] == 1)] = 1
            TP = torch.cumsum(TP, dim=0)
        
            map = torch.stack([TP/prec_num, TP/recall_num], dim=-1)
            map[torch.isnan(map)] = 0

            duplicate_mask = torch.cat((map[1:, 1] != map[:-1, 1], torch.tensor([True]).to(DEVICE)), dim=0)
            last_indices = torch.nonzero(duplicate_mask).squeeze().tolist()
            map = map[last_indices]
            map=torch.vstack((torch.tensor([0, 0]).to(DEVICE), map))
            map[:, 1] -= torch.cat((torch.tensor([0]).to(DEVICE), map[:-1,1]))
            ap = torch.sum(map[:, 0] * map[:, 1]).item()

            test_ap += ap

            test_acc += torch.sum(results[:,0] == results[:,1]).item()
            
            try:
                test_precision = true_positives / (true_positives + false_positives)
            except ZeroDivisionError:
                test_precision = 0
            try:
                test_recall = true_positives / (true_positives + false_negatives)
            except ZeroDivisionError:
                test_recall = 0
            pbar.set_description('Valid Error: | Loss: {:.4f} | Acc: {:.4f} | Prec: {:.4f} | Recall: {:.4f} | AP: {:.4f}'.format(test_loss/num_batches, test_acc/size, test_precision, test_recall, test_ap/num_batches))

    test_loss /= num_batches
    test_acc /= size
    test_ap /= num_batches

    return {'loss': test_loss, 'accuracy': test_acc, 'precision': test_precision, 'recall': test_recall, 'ap' : test_ap}

def load_sampler(dataset):
    
    label_list = dataset.label_list
    indices = dataset.indices
    num_true=0
    for index in indices:
        if label_list[index] == 1:
            num_true += 1
    num_false = len(indices) - num_true
    
    portion_true = num_true/len(indices)
    portion_false = num_false/len(indices)
    weights = [1/portion_true if label_list[index] == 1 else 1/portion_false for index in indices]
    
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return sampler

if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Get current file path
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.abspath(os.path.join(current_file_path, '..', 'config', 'config.yaml'))

    # Load configuation file
    with open(config_file,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    MODEL_NAME = config['network']["model_name"]
    model_path = os.path.abspath(os.path.join(current_file_path, '..',  'models', MODEL_NAME, 'model.pt'))

    # data
    pushnet_test_dataset = PushNetDataset(dataset_dir=config["data_path"], image_type=config['planner']['image_type'], type='test', zero_padding=config['file_zero_padding_num'])
    
    test_sampler = load_sampler(pushnet_test_dataset)
    test_dataloader = DataLoader(pushnet_test_dataset, 1000, test_sampler, num_workers=multiprocessing.cpu_count())

    # model
    model = PushNet()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path)) #
    
    test_metric = test_loop(test_dataloader, model, nn.CrossEntropyLoss())

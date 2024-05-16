#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.model import PushNet, NewModel
from utils.dataloader import PushNetDataset
import multiprocessing
import yaml

def train_loop(train_loader, model, loss_fn, optimizer):
    cur_batch = 0
    average_loss = 0
    average_acc = 0
    average_prec = 0
    average_recall = 0
    average_ap = 0
    model.train(True)
    pbar = tqdm(train_loader)
    
    for (images, velocities, labels) in pbar:
        images = images.to(DEVICE)
        velocities = velocities.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        pred = model(images, velocities)
        loss = loss_fn(pred, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print
        cur_batch += 1
        average_loss = average_loss + (loss.item()-average_loss)/cur_batch

        labels = torch.argmax(labels, dim=1) #(B, 2)

        results = torch.softmax(pred, dim=1)

        results = torch.stack([results[:,1], labels], dim=-1)
        results = results[torch.sort(results[:,0], descending=True)[1]]
        results = torch.where(results < 0.5, 0, 1)
    
        true_positives = torch.sum(torch.logical_and(results[:,0] == 1, results[:,1] == 1)).item()
        false_positives = torch.sum(torch.logical_and(results[:,0] == 1, results[:,1] == 0)).item()
        false_negatives = torch.sum(torch.logical_and(results[:,0] == 0, results[:,1] == 1)).item()

        recall_num = torch.tensor([true_positives + false_negatives]).repeat_interleave(results.shape[0]).to(DEVICE)
        prec_num = torch.zeros(results.shape[0]).to(DEVICE)
        prec_num[torch.where(results[:,0] == 1)] = 1
        prec_num = torch.cumsum(prec_num, dim=0)

        TP = torch.zeros(results.shape[0]).to(DEVICE)
        TP[torch.logical_and(results[:,0] == 1, results[:,1] == 1)] = 1
        TP = torch.cumsum(TP, dim=0)

        map = torch.stack([TP/prec_num, TP/recall_num], dim=-1)
        map[torch.isnan(map)] = 0

        duplicate_mask = torch.cat((map[1:, 1] != map[:-1, 1], torch.tensor([True]).to(DEVICE)), dim=-1)
        last_indices = torch.nonzero(duplicate_mask[:]).squeeze().tolist()
        map = map[last_indices,]
        map=torch.vstack((torch.tensor([0, 0]).to(DEVICE), map))
        map[:, 1] -= torch.cat((torch.tensor([0]).to(DEVICE), map[:-1,1]))
        ap = torch.sum(map[:, 0] * map[:, 1]).item()

        acc = torch.sum(results[:,0] == results[:,1]).item() / (results.shape[0])
        average_acc = average_acc + (acc-average_acc)/cur_batch
        average_ap = average_ap + (ap-average_ap)/cur_batch
        
        try:
            prec = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            prec = 0
        try:
            recall = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            recall = 0
        
        average_prec = average_prec + (prec-average_prec)/cur_batch
        average_recall = average_recall + (recall-average_recall)/cur_batch

        if cur_batch % 10 == 1:
            pbar.set_description('Train Error: | Loss: {:.4f} | Acc: {:.4f} | Prec: {:.4f} | Recall: {:.4f} | AP: {:.4f}'.format(average_loss, average_acc, average_prec, average_recall, average_ap))
    return {'loss': average_loss, 'accuracy': average_acc, 'precision': average_prec, 'recall': average_recall, 'ap' : average_ap}


def val_loop(val_loader, model, loss_fn):
    size = len(val_loader.dataset)
    num_batches = len(val_loader)

    val_loss, val_acc, val_ap = 0, 0 ,0
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
    model.train(False)
    pbar = tqdm(val_loader)    
    pbar.set_description('Valid Error: | Loss: {:.4f} | Acc: {:.4f} | Prec: {:.4f} | Recall: {:.4f} | AP: {:.4f}'.format(val_loss, val_acc, 0, 0, val_ap))
    with torch.no_grad():
        for images, velocities, labels in pbar:
            images = images.to(DEVICE)
            velocities = velocities.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            pred = model(images, velocities)
            loss = loss_fn(pred, labels)

            # Accumulate loss 
            val_loss += loss.item()

            # Accumulate accuracy
            labels = torch.argmax(labels, dim=1)
            results = torch.softmax(pred, dim=1)
            results = torch.stack([results[:,1], labels], dim=-1)
            results = results[torch.sort(results[:,0], descending=True)[1]]
            results = torch.where(results < 0.5, 0, 1)
            true_positives = torch.sum(torch.logical_and(results[:,0] == 1, results[:,1] == 1)).item()
            false_positives = torch.sum(torch.logical_and(results[:,0] == 1, results[:,1] == 0)).item()
            false_negatives = torch.sum(torch.logical_and(results[:,0] == 0, results[:,1] == 1)).item()

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

            val_ap += ap

            val_acc += torch.sum(results[:,0] == results[:,1]).item()
            
            tp_sum += true_positives
            fp_sum += false_positives
            fn_sum += false_negatives

            try:
                val_precision = tp_sum / (tp_sum + fp_sum)
            except ZeroDivisionError:
                val_precision = 0
            try:
                val_recall = tp_sum / (tp_sum + fn_sum)
            except ZeroDivisionError:
                val_recall = 0
            pbar.set_description('Valid Error: | Loss: {:.4f} | Acc: {:.4f} | Prec: {:.4f} | Recall: {:.4f} | AP: {:.4f}'.format(val_loss/num_batches, val_acc/size, val_precision, val_recall, val_ap/num_batches))

    val_loss /= num_batches
    val_acc /= size
    val_ap /= num_batches

    return {'loss': val_loss, 'accuracy': val_acc, 'precision': val_precision, 'recall': val_recall, 'ap' : val_ap}

def feature_extraction(dataloader,model, num_samples):
    print("Extracting features...")
    feature_map_size = 0
    features = np.zeros((1,128))
    images = np.zeros((1,1,96,96))
    pbar = tqdm(dataloader)
    
    with torch.no_grad():
        for image, pose, label in pbar:
            image = image.to(DEVICE)
            pose = pose.to(DEVICE)
            label = label.to(DEVICE)
            # Forward pass
            output = model(image, pose).cpu().numpy()
            features = np.concatenate((features, output), axis=0)
            images = np.concatenate((images, image.cpu()), axis=0)
            feature_map_size += 1
            if feature_map_size == num_samples:
                features = features[1:]
                images = images[1:]
                break
    return features, images

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
    
    # Get current file path
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.abspath(os.path.join(current_file_path, '..', 'config', 'config.yaml'))
    
    # Load configuation file
    with open(config_file,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Parameter settings
    # Data directories
    dataset_dir = config["data_dir"]
    tensor_dir = dataset_dir + '/tensors'
    # Image type
    image_type = config['planner']['image_type']
    # Train parameter
    zero_padding_num = config["file_zero_padding_num"]
    learning_rate = config["base_lr"] # 1e-4
    lr_decay_rate = config["decay_rate"] # 0.95
    weight_decay = config["train_l2_regularizer"] # 0.0005
    batch_size = config["batch_size"] # 64
    epochs = config["num_epochs"] # 60
    momentum_rate = config["momentum_rate"] # 0.9
    


    num_workers = multiprocessing.cpu_count()
    cur_date = datetime.today().strftime("%Y-%m-%d-%H%M")
    print('Starting training at {}. Device: {}'.format(cur_date, DEVICE))

    # data
    pushnet_train_dataset = PushNetDataset(dataset_dir, type='train', image_type=image_type, zero_padding=config["file_zero_padding_num"], pin_memory=False)
    pushnet_val_dataset = PushNetDataset(dataset_dir, type='val', image_type=image_type, zero_padding=config["file_zero_padding_num"], pin_memory=False)
    
    val_sampler = load_sampler(pushnet_val_dataset)
    train_sampler = load_sampler(pushnet_train_dataset)
    train_dataloader = DataLoader(pushnet_train_dataset, batch_size, train_sampler, num_workers=num_workers)
    val_dataloader = DataLoader(pushnet_val_dataset, 1000, val_sampler, num_workers=num_workers)

    # model
    model = PushNet()
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    # configure tensorboard
    dataset_dir = os.path.expanduser('~') + '/' + config["model_dir"]

    model_dir = os.path.abspath(os.path.join(os.path.expanduser('~'), config["model_dir"]))
    writer = SummaryWriter(model_dir + '/{}/logs'.format(cur_date))
    dataiter = iter(train_dataloader)
    images, poses, labels = next(dataiter)
    
    # Add a Projector to tensorboard
    feature_model= NewModel(model)
    feature_model.to(DEVICE)
    
    writer.add_graph(model, (images.to(DEVICE), poses.to(DEVICE)))
    writer.flush()
    
    epoch_start = 0
    min_val_loss = 100
    max_val_ap = 0
    
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model.parameters())
    for epoch in range(epoch_start,epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        train_metric = train_loop(train_dataloader, model, loss_fn, optimizer)
        val_metric = val_loop(val_dataloader, model, loss_fn)

        # write to tensorboard          
        loss_metric = {'train': train_metric['loss'], 'val': val_metric['loss']}
        acc_metric = {'train': train_metric['accuracy'], 'val': val_metric['accuracy']}
        precision_metric = {'train': train_metric['precision'], 'val': val_metric['precision']}
        recall_metric = {'train': train_metric['recall'], 'val': val_metric['recall']}
        ap_metric = {'train': train_metric['ap'], 'val': val_metric['ap']}

        writer.add_scalars('loss', loss_metric, epoch)
        writer.add_scalars('accuracy', acc_metric, epoch)
        writer.add_scalars('precision', precision_metric, epoch)
        writer.add_scalars('recall', recall_metric, epoch)
        writer.add_scalars('ap', ap_metric, epoch)
        
        writer.flush()

        if min_val_loss - val_metric['loss'] < -0.5:
            print('validation loss increase{}'.format(min_val_loss - val_metric['loss']))
            break
        if min_val_loss > val_metric['loss']:
            torch.save(model.state_dict(), model_dir + '/{}/'.format(cur_date) + 'model' + str(epoch + 1) + '_loss_' + str(val_metric['loss']) +'.pt')
            min_val_loss = val_metric['loss']
        elif max_val_ap < val_metric['ap']:
            torch.save(model.state_dict(), model_dir + '/{}/'.format(cur_date) + 'model' + str(epoch + 1) + '_ap_' + str(val_metric['ap']) +'.pt')
            max_val_ap = val_metric['ap']

        print('-'*10)
    torch.save(model.state_dict(), model_dir + '/{}/'.format(cur_date) + 'model_trained_' + str(val_metric['loss']) +'.pt')
    

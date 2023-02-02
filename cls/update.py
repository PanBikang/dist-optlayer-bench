#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.optim as optim

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return image.clone().detach().requires_grad_(True), label.clone().detach().requires_grad_(True)
        return torch.as_tensor(image), torch.as_tensor(label)
        # return Variable(image), Variable(label)


class LocalUpdate(object):
    def __init__(self, config_dict, dataset, idxs, logger, device):
        self.config_dict = config_dict
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = device
        # Default criterion set to NLL loss function
        # but now useless
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.config_dict['batchSz'], shuffle=True, **kwargs)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.config_dict['batchSz'], shuffle=False, **kwargs)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.config_dict['batchSz'], shuffle=False, **kwargs)
        return trainloader, validloader, testloader

    def update_weights(self, model, epoch, optimizer, trainF):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates

        for iter_cnt in range(self.config_dict['local_ep']):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = model(images)
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                pred = log_probs.data.max(1)[1]
                incorrect = pred.ne(labels.data).cpu().sum()
                err = 100.*incorrect/len(images)
                if self.config_dict['verbose'] and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, iter_cnt, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                trainF.write('{},{},{:.6f},{}\n'.format(epoch, iter_cnt, loss.data, err))
                trainF.flush()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, epoch, user_id, valF):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = F.nll_loss(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        valF.write('{},{},{},{}\n'.format(epoch, user_id, loss, 1 - accuracy))
        valF.flush()
        return accuracy, loss
    
    


def test_inference(config_dict, model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=config_dict['batchSz'],
                            shuffle=False, **kwargs)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

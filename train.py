######
# Useful training functions and classes
######

from tqdm.notebook import trange, tqdm
import time
import pickle
from threading import Thread

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import os
import glob
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from util import *

class TrainModel():
    def __init__(self, model):
        self.model = model
        self.was_training = model.training

    def __enter__(self):
        self.model.train()

    def __exit__(self, type, value, traceback):
        self.model.train(mode=self.was_training)

class EvalModel():
    def __init__(self, model):
        self.model = model
        self.was_training = model.training

    def __enter__(self):
        self.model.eval()

    def __exit__(self, type, value, traceback):
        self.model.train(mode=self.was_training)

class TrainingCallback():
    def on_train_begin(self, model, optimizer, schedule, cb_dict): pass
    def on_epoch_begin(self, model, optimizer, schedule, cb_dict): pass
    def on_batch_begin(self, model, optimizer, schedule, cb_dict): pass
    def on_batch_end(self, model, optimizer, schedule, cb_dict, loss): pass
    def on_epoch_end(self, model, optimizer, schedule, cb_dict): pass
    def on_train_end(self, model, optimizer, schedule, cb_dict): pass
    def state_dict(self): pass
    def load_state_dict(self, state_dict): pass

class LogLoss(TrainingCallback):
    def __init__(self, metric_name = "Training Loss"):
        self.metric_name = metric_name

    def on_epoch_begin(self, model, optimizer, schedule, cb_dict):
        self.running_total = 0
        self.num_examples = 0
        cb_dict[self.metric_name] = None

    def on_batch_end(self, model, optimizer, schedule, cb_dict, loss):
        self.running_total += loss
        self.num_examples += 1
    
    def on_epoch_end(self, model, optimizer, schedule, cb_dict):
        cb_dict[self.metric_name] = self.running_total / self.num_examples
        print(f"Training loss: {self.running_total / self.num_examples}")

class Visualize(TrainingCallback):
    def __init__(self, dataloader, denorm_fn):
        self.denorm_fn = denorm_fn
        self.im1, self.im2, self.target = next(iter(dataloader))

    def on_epoch_end(self, model, optimizer, schedule, cb_dict):
        im1, im2 = self.im1.to('cuda'), self.im2.to('cuda')

        with EvalModel(model): 
            preds = predict_flow(model, im1, im2)

        for im1, im2, pred, target in zip(self.im1, self.im2, preds, self.target):
            im1 = self.denorm_fn(im1)
            im1 = im1.numpy().transpose((1, 2, 0))

            im2 = self.denorm_fn(im2)
            im2 = im2.numpy().transpose((1, 2, 0))

            pred = flow2rgb(pred.detach().cpu().numpy())
            target = flow2rgb(target.numpy())

            fig, ax = plt.subplots(1, 4, figsize=(12, 12))

            ax[0].imshow(im1)
            ax[1].imshow(im2)
            ax[2].imshow(target)
            ax[3].imshow(pred)

            plt.show()

class LRSchedule(TrainingCallback):
    def __init__(self, schedule):
        self.schedule = schedule

    def on_batch_end(self, *args, **kwargs):
        self.schedule.step()

class Checkpoint(TrainingCallback):
    def __init__(self, ckpt_file, interval=5*60, reset=False):
        self.ckpt_file = ckpt_file
        if not self.ckpt_file.endswith('.ckpt.tar'): self.ckpt_file += '.ckpt.tar'
        self.interval = interval
        self.reset = reset

    def load(self, model, optimizer, schedule):
        if not os.path.exists(self.ckpt_file): return

        print("\n--- LOADING CHECKPOINT ---")

        checkpoint = torch.load(self.ckpt_file, map_location=None)

        if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
        if 'schedule' in checkpoint: schedule.load_state_dict(checkpoint['schedule'])

    def _checkpoint(self, model, optimizer, schedule):
        state = {
            'model': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'schedule': schedule.state_dict(),
        }

        torch.save(state, self.ckpt_file)

    def checkpoint(self, model, optimizer, schedule):
        # Save model in a separate thread (otherwise interrupting training during a save corrupts the checkpoint file)
        a = Thread(target=self._checkpoint, args=(model, optimizer, schedule)) 
        a.start()
        a.join()

    def on_train_begin(self, model, optimizer, schedule, cb_dict):
        if not self.reset: self.load(model, optimizer, schedule)
        self.start_time = time.time()      

    def on_batch_end(self, model, optimizer, schedule, cb_dict, loss):
        end = time.time()
        elapsed = end - self.start_time

        if elapsed > self.interval:
            self.start_time = end
            self.checkpoint(model, optimizer, schedule)
            print("\n--- CHECKPOINT ---")

class TrainingSchedule():
    def __init__(self, dataloader, num_epochs, callbacks):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epoch = 0
        self.iteration = 0
        self.callbacks = callbacks
        self.cb_dict = {}

    def data(self):
        self.iteration = 0

        for data in tqdm(self.dataloader, desc=f"Epoch {self.epoch+1}", leave=False):
            self.iteration += 1
            yield data

    def __iter__(self):
        for i in trange(self.epoch, self.num_epochs, initial=self.epoch, total=self.num_epochs):
            self.epoch = i
            yield i

    def state_dict(self):
        callbacks_state = [callback.state_dict() for callback in self.callbacks]
        return pickle.dumps({'callbacks': callbacks_state, 'cb_dict': self.cb_dict, 'epoch': self.epoch, 'iteration': self.iteration})

    def load_state_dict(self, state_dict):
        state_dict = pickle.loads(state_dict)

        for cb, cb_state_dict in zip(self.callbacks, state_dict['callbacks']): 
            cb.load_state_dict(cb_state_dict)

        self.epoch = state_dict['epoch']
        self.iteration = state_dict['iteration']
        self.cb_dict = state_dict['cb_dict']

    def on_train_begin(self, model, optimizer):        
        for cb in self.callbacks:
            cb.on_train_begin(model, optimizer, self, self.cb_dict)

    def on_epoch_begin(self, model, optimizer):              
        for cb in self.callbacks:
            cb.on_epoch_begin(model, optimizer, self, self.cb_dict)

    def on_batch_begin(self, model, optimizer): 
        for cb in self.callbacks:
            cb.on_batch_begin(model, optimizer, self, self.cb_dict)

    def on_batch_end(self, model, optimizer, loss): 
        for cb in self.callbacks:
            cb.on_batch_end(model, optimizer, self, self.cb_dict, loss)
            
    def on_epoch_end(self, model, optimizer): 
        for cb in self.callbacks:
            cb.on_epoch_end(model, optimizer, self, self.cb_dict)

    def on_train_end(self, model, optimizer): 
        for cb in self.callbacks:
            cb.on_train_end(model, optimizer, self, self.cb_dict)

def train(model, optimizer, objective, schedule):
    cb_dict = {}

    with TrainModel(model):
        schedule.on_train_begin(model, optimizer)

        for epoch in schedule:

            schedule.on_epoch_begin(model, optimizer)

            for im1, im2, target in schedule.data():
                schedule.on_batch_begin(model, optimizer)

                im1, im2, target = Variable(im1.to('cuda')), Variable(im2.to('cuda')), Variable(target.to('cuda'))

                out = model(im1, im2)
                loss = objective(out, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                schedule.on_batch_end(model, optimizer, loss.detach().item())

                del loss, im1, im2, target

            schedule.on_epoch_end(model, optimizer)

        schedule.on_train_end(model, optimizer)
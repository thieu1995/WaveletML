#!/usr/bin/env python
# Created by "Thieu" at 03:51, 19/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import os
import torch
from torch.utils.tensorboard import SummaryWriter


class BaseCallback:
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_batch_begin(self, batch, logs=None): pass
    def on_batch_end(self, batch, logs=None): pass
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass


class PrintLossCallback(BaseCallback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"[Epoch {epoch+1}] Loss: {logs['loss']:.4f}")


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs['loss']
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                self.stop_training = True


class ModelCheckpointCallback(BaseCallback):
    def __init__(self, save_path="best_model.pt", monitor="loss", mode="min"):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs[self.monitor]
        if (self.mode == "min" and current_score < self.best_score) or \
           (self.mode == "max" and current_score > self.best_score):
            self.best_score = current_score
            torch.save(logs['model_state_dict'], self.save_path)
            print(f"Saved model at epoch {epoch+1} with {self.monitor}: {current_score:.4f}")


class TensorBoardLoggerCallback(BaseCallback):
    def __init__(self, log_dir="runs/wavelet_experiment"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        self.writer.add_scalar("Loss/train", logs["loss"], epoch)

    def on_train_end(self, logs=None):
        self.writer.close()

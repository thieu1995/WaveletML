#!/usr/bin/env python
# Created by "Thieu" at 03:51, 19/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch


class BaseCallback:
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_batch_begin(self, batch, logs=None): pass
    def on_batch_end(self, batch, logs=None): pass
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass


class PrintLossCallback(BaseCallback):
    def on_epoch_end(self, epoch, logs=None):
        msg = f"[Epoch {epoch+1}] Loss: {logs['loss']:.6f}"
        if logs.get("val_loss") is not None:
            msg += f" | Val Loss: {logs['val_loss']:.6f}"
        print(msg)


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=5, min_delta=1e-4, monitor="val_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = float('inf')
        self.counter = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        score = logs.get(self.monitor)
        if score is None:
            return
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                self.stop_training = True


class ModelCheckpointCallback(BaseCallback):
    def __init__(self, save_path="best_model.pt", monitor="val_loss", mode="min"):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        score = logs.get(self.monitor)
        if score is None:
            return
        if (self.mode == "min" and score < self.best_score) or \
           (self.mode == "max" and score > self.best_score):
            self.best_score = score
            torch.save(logs["model_state_dict"], self.save_path)
            print(f"Saved model at epoch {epoch+1} with {self.monitor}: {score:.4f}")


class FileLoggerCallback(BaseCallback):
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file
        with open(self.log_file, "w") as f:
            f.write("epoch,loss,val_loss\n")

    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_file, "a") as f:
            line = f"{epoch},{logs.get('loss')},{logs.get('val_loss', '')}\n"
            f.write(line)

    def on_train_end(self, logs=None):
        print(f"Training log saved to {self.log_file}")

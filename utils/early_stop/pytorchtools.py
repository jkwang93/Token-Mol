# _*_ encoding: utf-8 _*_
__author__ = 'wjk'
__date__ = '2019/12/18 16:07'
'''https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py'''

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        # self.model_name = model_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 100
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, model_name):

        # score = -val_loss
        score = val_loss


        if self.best_score == 100:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
            print('self.best_score:',self.best_score)

        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print('best_score:',self.best_score, 'now:',score)
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        print('now best_score:', self.best_score)

        # model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.save_pretrained(model_name)

        torch.save(model.state_dict(), model_name + '_early_stop.pt')



        self.val_loss_min = val_loss

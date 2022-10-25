# -*- coding: utf-8 -*-
import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import datetime, pickle
import os
import numpy as np
import random
from torch import Tensor

def renormalize(predicted_shot, max_thres, epsilon=0.01, clip=True, directed=True):
    predicted_shot = predicted_shot * max_thres
    if clip:
        if not directed:
            predicted_shot = (predicted_shot + predicted_shot.transpose(1, 2)) / 2
        for j in range(predicted_shot.size(-1)):
            predicted_shot[:, j, j] = 0
        mask = predicted_shot >= epsilon
        predicted_shot = predicted_shot * mask.float()
    else:
        pass
    return predicted_shot

def JS_divergence(p, q, tensor_mode=False):
    '''
    input: (batch size * |V| * |V|) or (|V| * |V|)
    '''
    if len(p.size()) > 2:
        batch_size = p.size(0)
        JS = 0
        for s in range(batch_size):
            p_ = (p[s, :] / p[s, :].sum())
            q_ = (q[s, :] / q[s, :].sum())
            JS += JS_divergence_tensor(p_, q_)
        return JS / batch_size
    else:
        p_ = (p / p.sum())
        q_ = (q / q.sum())
        return JS_divergence_tensor(p_, q_, tensor_mode)

def JS_divergence_tensor(p, q, tensor_mode=False):
    M = (p + q) / 2
    p_mask = p > 0
    q_mask = q > 0
    JS = 0.5 * (p.masked_select(p_mask) * torch.log((p / M).masked_select(p_mask))).sum() + 0.5 * (q.masked_select(q_mask) * torch.log((q / M).masked_select(q_mask))).sum()
    if tensor_mode:
        return JS
    else:
        return JS.item()

def RMSE(input, target):
    '''
    input: batch size * |V| * |V|
    '''
    if len(input.size()) > 2:
        batch_size = input.size(0)
        num_items = input.numel() / batch_size
        f_norm_sum = 0
        for s in range(batch_size):
            f_norm_sum = f_norm_sum + torch.sqrt((input[s, :].cpu() - target[s, :].cpu()).pow(2).sum() / num_items)
        return f_norm_sum.item() / batch_size
    else:
        '''
        input: |V| * |V|
        '''
        num_items = input.numel()
        return torch.sqrt((input.cpu() - target.cpu()).pow(2).sum() / num_items).item()

'''
Lei K, Qin M, Bai B, et al. Gcn-gan: A non-linear temporal link prediction model for weighted dynamic networks[C]
//IEEE INFOCOM 2019-IEEE Conference on Computer Communications. IEEE, 2019: 388-396.
'''
def MissRate(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask1 = (input > 0) & (target == 0)
    mask2 = (input == 0) & (target > 0)
    mask = mask1 | mask2
    return mask.sum().item() / num

def balanced_sample(y_pred, y_true):
    num_neg_samples = int((y_true > 0).float().sum().item())
    prob = 1. - num_neg_samples / y_true.numel()  # Probability to sample a negative.

    sample_size = int(1.1 * num_neg_samples / prob)  # (Over)-sample size.

    neg_idx = None
    mask = (y_true == 0.)
    for _ in range(3):  # Number of tries to sample negative indices.
        rnd = sample(y_true.numel(), sample_size, y_true.device)
        try:
            rnd = rnd[mask[rnd]]  # Filter true negatives.
        except:
            print()
        neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
        if neg_idx.numel() >= num_neg_samples:
            neg_idx = neg_idx[:num_neg_samples]
            break
        mask[neg_idx] = False
    mask[neg_idx] = False
    mask = ~mask
    return y_pred.masked_select(mask), y_true.masked_select(mask)

def sample(population: int, k: int, device=None) -> Tensor:
    if population <= k:
        return torch.arange(population, device=device)
    else:
        return torch.tensor(random.sample(range(population), k), device=device)

def get_auc(y_pred, y_true, balance=True):
    if len(y_pred.size()) > 2:
        batch_size = y_pred.size(0)
        total_roc_auc = 0
        for s in range(batch_size):
            y_pred_batch = y_pred[s, :].cpu()
            y_true_batch = y_true[s, :].cpu()
            if len(y_pred_batch.shape) > 1:
                y_true_batch = y_true_batch.view(-1)
                y_pred_batch = y_pred_batch.view(-1)
            if balance:
                y_pred_batch, y_true_batch = balanced_sample(y_pred_batch, y_true_batch)
            roc_auc = roc_auc_score(y_true_batch, y_pred_batch)
            total_roc_auc += roc_auc
        ave_roc_auc = total_roc_auc / batch_size
        return ave_roc_auc
    else:
        y_pred_test = y_pred.cpu().view(-1)
        y_true_test = y_true.cpu().view(-1)
        if balance:
            y_pred_test, y_true_test = balanced_sample(y_pred_test, y_true_test)
        roc_auc = roc_auc_score(y_true_test, y_pred_test)
        return roc_auc

def diag_zeros(A):
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    if A.dim() == 3:
        for i in range(A.size(0)):
            A[i] -= torch.diag_embed(diag[i])
    elif A.dim() == 2:
        A -= torch.diag_embed(diag)
    return A

class EvalDataRecorder(object):
    def __init__(self):
        self.record = defaultdict(list)
        self.weight_indicator = {'RMSE': RMSE, 'JS': JS_divergence, 'mismatch': MissRate}
        self.link_indicator = {'AUROC': get_auc}

    def update_weight_eval(self, input, target, printf=True, save_data=True, indicator=None):
        if save_data:
            self.record['weight_pred'].append(input.detach().cpu())
            self.record['weight_true'].append(target.detach().cpu())
        input = diag_zeros(input)
        target = diag_zeros(target)
        batch_size = input.size(0)
        if indicator is None:
            for i in self.weight_indicator:
                self.record[i].append(batch_size * self.weight_indicator[i](input, target))
                if printf:
                    print('{}: {:.4f}'.format(i, self.record[i][-1] / batch_size))
        elif indicator in self.weight_indicator:
            self.record[indicator].append(batch_size * self.weight_indicator[indicator](input, target))
            if printf:
                print('{}: {:.4f}'.format(indicator, self.record[indicator][-1] / batch_size))
        else:
            raise Exception('Unsupported indicator: {}'.format(indicator))

    def update_link_eval(self, input, target, printf=True, save_data=True, indicator=None):
        if save_data:
            self.record['link_pred'].append(input.detach().cpu())
            self.record['link_true'].append(target.detach().cpu())
        input = diag_zeros(input)
        target = diag_zeros(target)
        batch_size = input.size(0)
        if indicator is None:
            for i in self.link_indicator:
                self.record[i].append(batch_size * self.link_indicator[i](input, target))
                if printf:
                    print('{}: {:.4f}'.format(i, self.record[i][-1] / batch_size))
        elif indicator in self.link_indicator:
            self.record[indicator].append(batch_size * self.link_indicator[indicator](input, target))
            if printf:
                print('{}: {:.4f}'.format(indicator, self.record[indicator][-1] / batch_size))
        else:
            raise Exception('Unsupported indicator: {}'.format(indicator))

    def add_ave_eval(self, batch_size, printf=True):
        for indicator in self.weight_indicator:
            self.record['ave_{}'.format(indicator)].append(sum(self.record[indicator]) / (batch_size * len(self.record[indicator])))
            if printf:
                print('ave_{}: {:.4f}'.format(indicator, self.record['ave_{}'.format(indicator)][-1]))
        for indicator in self.link_indicator:
            self.record['ave_{}'.format(indicator)].append(sum(self.record[indicator]) / (batch_size * len(self.record[indicator])))
            if printf:
                print('ave_{}: {:.4f}'.format(indicator, self.record['ave_{}'.format(indicator)][-1]))
        if 'running_time' in self.record:
            self.record['ave_running_time'].append(np.mean(self.record['running_time']))
            if printf:
                print('ave_running_time: {:.4f}'.format(self.record['ave_running_time'][-1]))

    def update_running_time(self, running_time):
        self.record['running_time'].append(running_time)

    def save_record(self, model, dataset, config, save_dir=None):
        if save_dir is None:
            save_dir = os.path.realpath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../results'))
        dt = datetime.datetime.now()
        date = f"{dt.year}_{dt.month}_{dt.day}"
        save_dir = os.path.join(save_dir, model, dataset, date)
        os.makedirs(save_dir, exist_ok=True)
        save_dict = {"config": config, "result": self.record}
        pickle.dump(save_dict, open(os.path.join(save_dir, 'result.pkl'), 'wb'), protocol=4)










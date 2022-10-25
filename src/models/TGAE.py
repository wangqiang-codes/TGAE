# -*- coding: utf-8 -*-
'''
@Time    : 2022/5/7 10:06
@Author  : Wang Qiang
@FileName: TGAE.py
'''
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import sys
import os
sys.path.append(os.path.split(os.path.realpath(__file__))[0])
sys.path.append(os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")))
from utils import setup_seed, EvalDataRecorder, JS_divergence
from utils.train_utils import add_flags_from_config
setup_seed(0)
import numpy as np
np.random.seed(0)
from layers import DirectedGNNEncoder, DirectedInnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
import scipy.sparse as sp
import argparse
import time
# from torch.utils.tensorboard import SummaryWriter
from utils.eval_utils import EvalDataRecorder, renormalize
from scipy.optimize import dual_annealing

try:
    from geomloss import SamplesLoss
except:
    print('geomloss has not been installed successfully.')

EPS = 1e-15

def extract_indices_values(adj):
    device = adj.device
    coo = sp.coo_matrix(adj.cpu().numpy())
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    values = coo.data
    values = torch.from_numpy(values).float()
    return indices.to(device), values.to(device)

def adj_to_edgelist(adj_graphs):
    '''
    batch * snapshots * nodes * nodes
    '''
    batch_size, window_size, node_num = adj_graphs.size()[0: 3]
    edges_list = []
    for batch in range(batch_size):
        edges_list_ = [extract_indices_values(adj_graph) for adj_graph in adj_graphs[batch]]
        edge_index_list = [x[0] for x in edges_list_]
        edge_weight_list = [x[1] for x in edges_list_]
        edges_list.append((edge_index_list, edge_weight_list))
    return edges_list

class TGAE:
    def __init__(self, device, config):
        parser = argparse.ArgumentParser()
        parser = add_flags_from_config(parser, config)
        args = parser.parse_args()
        self.args = args
        self.args.device = device
        self.args.patience = self.args.epochs if not self.args.patience else int(self.args.patience)

        if self.args.feature == 'adj':
            self.args.feat_dim = self.args.node_num
        else:
            self.args.feat_dim = 2
        self.args.weight_dim = 1
        self.args.nb_nodes = self.args.node_num
        self.model = TGAEmodeler(self.args).to(self.args.device)
        if self.args.optimizer == 'RMSprop':
            self.optimiser = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            self.optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=self.args.lr, weight_decay=self.args.weight_decay)
        if not self.args.lr_reduce_freq:
            self.args.lr_reduce_freq = self.args.epochs
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=int(self.args.lr_reduce_freq),
            gamma=float(self.args.gamma)
        )

        self.batch = 0
        self.feature = None

        self.args.save_dir = os.path.realpath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../saved_model'))
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)


    def train(self, train_in_shots, train_out_shot):
        if self.args.SummaryWriter:
            # Writer will output to ./runs/ directory by default
            # writer = SummaryWriter()
            edr = EvalDataRecorder()

        print('\ntrain TGAE on {}...'.format(self.args.dataset))
        device = self.args.device
        train_in_shots, train_out_shot = train_in_shots.to(device), train_out_shot.to(device)

        train_out_shot_weight = torch.cat([train_in_shots, train_out_shot.view(train_out_shot.size(0), -1, self.args.node_num, self.args.node_num)], dim=1)[:, 1:, :, :]

        train_out_shot = train_out_shot.view(train_out_shot.size(0), -1)

        if self.args.feature == 'adj':
            self.feature = train_in_shots
        else:
            self.feature = torch.cat([train_in_shots.sum(dim=-1).unsqueeze(-1) + train_in_shots.sum(dim=-2).unsqueeze(-1),
                                      (train_in_shots > 0).float().sum(dim=-1).unsqueeze(-1) + (train_in_shots > 0).float().sum(dim=-2).unsqueeze(-1)], dim=-1)

        train_in_shots_edgelist = adj_to_edgelist(train_in_shots)
        train_out_shot_weight_edgelist = adj_to_edgelist(train_out_shot_weight)

        start = time.time()
        # train
        best = float('inf')
        best_t = 0
        cnt_wait = 0
        for epoch in range(self.args.epochs):

            self.model.train()
            self.optimiser.zero_grad()
            decode_result, loss = self.model(self.feature, train_in_shots_edgelist, train_out_shot_weight_edgelist)

            if epoch % 10 == 0:
                print('[epoch %d] [loss %.4f]' % (epoch, loss.item()))
            if loss < best:
                best = loss
                cnt_wait = 0
                best_t = epoch
                if self.args.save:
                    torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, 'temp', 'best_{}_{}_ProcessID_{}.pth'.format(self.args.dataset,
                                                                                                                                 'TGAE',
                                                                                                                                 self.args.processID)))
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience and epoch > self.args.min_epochs:
                print('Early stopping! (best epoch: {})'.format(best_t))
                break

            if self.args.SummaryWriter and epoch % self.args.log_freq == 0:

                # writer.add_scalar('Loss/train(batch: {})'.format(self.batch), loss.item(), epoch)
                self.model.eval()
                # link prediction evaluation
                train_label = (train_out_shot > 0.).float().cpu().detach().view(-1, self.args.nb_nodes, self.args.nb_nodes)
                predicted_shot = decode_result['out_shot_weight'].cpu().detach().view(-1, self.args.nb_nodes, self.args.nb_nodes)
                edr.update_link_eval(predicted_shot, train_label, printf=False)
                edr.update_gmauc(predicted_shot, train_label, train_in_shots.detach().cpu(), train_out_shot.view(-1, self.args.nb_nodes, self.args.nb_nodes).detach().cpu(), printf=False)

                # link prediction evaluation (weight)
                predicted_shot = renormalize(predicted_shot, self.args.max_thres, epsilon=0.01)
                edr.update_weight_eval(predicted_shot, (train_out_shot * self.args.max_thres).cpu().view(-1, self.args.nb_nodes, self.args.nb_nodes), printf=False)

                # for indicator in edr.weight_indicator:
                #     writer.add_scalar('{}/train(batch: {})'.format(indicator, self.batch), edr.record[indicator][-1], epoch)
                # for indicator in edr.link_indicator:
                #     writer.add_scalar('{}/train(batch: {})'.format(indicator, self.batch), edr.record[indicator][-1], epoch)

            loss.backward()
            if self.args.grad_clip is not None:
                for param in list(self.model.parameters()):
                    nn.utils.clip_grad_norm_(param, self.args.grad_clip)
            self.optimiser.step()
            self.lr_scheduler.step()

        # if self.args.SummaryWriter:
        #     writer.close()

        # load model
        if self.args.save:
            print('Loading {}th epoch'.format(best_t))
            self.model.load_state_dict(torch.load(os.path.join(self.args.save_dir, 'temp',
                                                          'best_{}_{}_ProcessID_{}.pth'.format(self.args.dataset,
                                                                                               'TGAE',
                                                                                               self.args.processID))))
        end = time.time()
        print('time: {:.2f}s'.format(end - start))
        del train_in_shots, train_out_shot
        return end - start

    def train_offline(self, data, train_shots):
        print('\ntrain TGAE on {}...'.format(self.args.dataset))
        device = self.args.device

        start = time.time()
        # train
        best = float('inf')
        best_t = 0
        cnt_wait = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_losses = []
            for t in train_shots:
                train_in_shots, train_out_shot = data[t]
                train_in_shots, train_out_shot = train_in_shots.to(device), train_out_shot.to(device)

                if self.args.step_prediction:
                    train_out_shot_weight = torch.cat([train_in_shots, train_out_shot.view(train_out_shot.size(0), -1,
                                                                                           self.args.node_num,
                                                                                           self.args.node_num)], dim=1)[
                                            :, 1:, :, :]
                else:
                    train_out_shot_weight = train_out_shot.view(train_out_shot.size(0), -1, self.args.node_num,
                                                                self.args.node_num)

                if self.args.feature == 'adj':
                    self.feature = train_in_shots
                else:
                    self.feature = torch.cat(
                        [train_in_shots.sum(dim=-1).unsqueeze(-1) + train_in_shots.sum(dim=-2).unsqueeze(-1),
                         (train_in_shots > 0).float().sum(dim=-1).unsqueeze(-1) + (train_in_shots > 0).float().sum(
                             dim=-2).unsqueeze(-1)], dim=-1)

                train_in_shots_edgelist = adj_to_edgelist(train_in_shots)
                train_out_shot_weight_edgelist = adj_to_edgelist(train_out_shot_weight)

                self.optimiser.zero_grad()
                decode_result, loss = self.model(self.feature, train_in_shots_edgelist, train_out_shot_weight_edgelist)

                loss.backward()
                if self.args.grad_clip is not None:
                    for param in list(self.model.parameters()):
                        nn.utils.clip_grad_norm_(param, self.args.grad_clip)
                self.optimiser.step()
                self.lr_scheduler.step()
                epoch_losses.append(loss.item())

            average_loss = sum(epoch_losses) / len(epoch_losses)
            if epoch % 10 == 0:
                print('[epoch %d] [loss %.4f]' % (epoch, average_loss))
            if average_loss < best:
                best = average_loss
                cnt_wait = 0
                best_t = epoch
                if self.args.save:
                    torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, 'temp', 'best_{}_{}_ProcessID_{}.pth'.format(self.args.dataset,
                                                                                                                                 'TGAE',
                                                                                                                                 self.args.processID)))
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience and epoch > self.args.min_epochs:
                print('Early stopping! (best epoch: {})'.format(best_t))
                break

        # load model
        if self.args.save:
            print('Loading {}th epoch'.format(best_t))
            self.model.load_state_dict(torch.load(os.path.join(self.args.save_dir, 'temp',
                                                          'best_{}_{}_ProcessID_{}.pth'.format(self.args.dataset,
                                                                                               'TGAE',
                                                                                               self.args.processID))))
        end = time.time()
        print('time: {:.2f}s'.format(end - start))
        return end - start

    def test(self, test_in_shots):
        # test
        start = time.time()
        test_in_shots = test_in_shots.to(self.args.device)
        if self.args.feature == 'adj':
            self.feature = test_in_shots
        else:
            self.feature = torch.cat([test_in_shots.sum(dim=-1).unsqueeze(-1) + test_in_shots.sum(dim=-2).unsqueeze(-1),
                                      (test_in_shots > 0).float().sum(dim=-1).unsqueeze(-1) + (test_in_shots > 0).float().sum(dim=-2).unsqueeze(-1)], dim=-1)
        self.model.eval()
        decode_result, _ = self.model(self.feature, adj_to_edgelist(test_in_shots))
        end = time.time()
        result = {}
        result['running_time'] = end - start
        result['out_shot_link'] = torch.stack([decode_result[i]['out_shot_link'].detach() for i in decode_result])
        result['out_shot_weight'] = torch.stack([decode_result[i]['out_shot_weight'].detach() for i in decode_result])
        return result

    def embed(self, adj):
        adj_in_shots = adj.to(self.args.device)
        self.model.eval()
        embedding, _, _ = self.model.encode(self.feature, adj_in_shots)
        return embedding.detach().cpu()

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def best_threshold(pred, true):
    def fun(x):
        return ((pred > x[0]).float() - true).abs().sum().item()
    bounds = [[0, 1]]
    res = dual_annealing(fun, bounds)
    return res.x.item()

class TGAEmodeler(nn.Module):
    def __init__(self, args):
        super(TGAEmodeler, self).__init__()
        self.args = args

        self.encoder = DirectedGNNEncoder(args)

        self.decoder = DirectedInnerProductDecoder()
        self.link_act = 'sigmoid'
        self.weight_act = 'sigmoid'
        self.lstm = nn.LSTM(
            input_size=args.dim * args.nb_nodes * 2,
            hidden_size=args.lstm_dim,
            num_layers=1,
            batch_first=True,
            dropout=args.dropout
        )

        self.lin_weight_s = nn.Linear(args.lstm_dim, args.nb_nodes * args.dim)
        self.lin_weight_t = nn.Linear(args.lstm_dim, args.nb_nodes * args.dim)

        self.lin_link_s = nn.Linear(args.lstm_dim, args.nb_nodes * args.dim)
        self.lin_link_t = nn.Linear(args.lstm_dim, args.nb_nodes * args.dim)
        self.reset_parameters()

        if self.args.loss_func == 'L1loss':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif self.args.loss_func == 'mse':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise Exception('Unsupported weight loss function: {}'.format(self.args.loss_func))

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        reset(self.lin_weight_s)
        reset(self.lin_weight_t)
        reset(self.lin_link_s)
        reset(self.lin_link_t)

    def forward(self, feature, train_in_shots, train_out_shots=None):
        '''
        :param feature: FloatTensor (batch_size, window_size, node_num, feature_dim)
        :param train_in_shots: list (batch_size, 2<edge list, weight list>, snapshots, edges<weights>)
        :param train_out_shots: list (batch_size, 2<edge list, weight list>, snapshots, edges<weights>)
        :return result: {'loss': , 'out_shot': }
        '''
        embeddings = self.encode(feature, train_in_shots)
        batch_size, window_size = len(train_in_shots), len(train_in_shots[0][0])
        encoder_output = embeddings.view(batch_size, window_size, -1)
        h_seq, (hn, _) = self.lstm(encoder_output)

        if train_out_shots is not None:
            decode_result = self.decode(h_seq, hn, training=True)
            loss = self.recon_loss(decode_result, train_out_shots)
            return decode_result, loss
        else:
            decode_result = self.decode(h_seq, hn, training=False)
            if self.args.weight_coef != 1:
                batch_size, window_size, node_num = feature.size()[0: 3]
                for batch in range(batch_size):
                    target = torch.zeros(window_size - 1, node_num, node_num)
                    for j in range(1, window_size):
                        target[j - 1, train_in_shots[batch][0][j][0], train_in_shots[batch][0][j][1]] = 1.0
                    self.args.epsilon = best_threshold(decode_result[batch]['link_pred'].to(target.device), target)
                    print('best threshold: {:.4f}'.format(self.args.epsilon))
                    decode_result[batch]['out_shot_weight'] *= (decode_result[batch]['out_shot_link'] > self.args.epsilon).float()

            return decode_result, None

    def encode(self, feature, train_in_shots):
        batch_size, window_size, node_num = feature.size()[0: 3]

        embeddings = None
        for i in range(batch_size):
            embeddings_batch = None
            for j in range(window_size):
                # next timestamp
                feature_graph = feature[i, j, :]

                edge_index, edge_weight = train_in_shots[i][0][j], train_in_shots[i][1][j]

                embedding_s, embedding_t = self.encoder(feature_graph, feature_graph, edge_index, edge_weight)
                # concat source embeddings and target embeddings
                embedding = torch.cat([embedding_s, embedding_t], dim=-1)
                if embeddings_batch is None:
                    embeddings_batch = embedding.unsqueeze(0)
                else:
                    embeddings_batch = torch.cat((embeddings_batch, embedding.unsqueeze(0)))

            if embeddings is None:
                embeddings = embeddings_batch.unsqueeze(0)
            else:
                embeddings = torch.cat((embeddings, embeddings_batch.unsqueeze(0)))
        return embeddings

    def decode(self, lstm_h_seq, lstm_hn, training=True):
        result = {}
        for batch in range(lstm_h_seq.size(0)):
            result[batch] = {}
            if not training:
                out_shot_weight_s = self.lin_weight_s(lstm_hn[batch]).view(self.args.nb_nodes, -1)
                out_shot_weight_t = self.lin_weight_t(lstm_hn[batch]).view(self.args.nb_nodes, -1)

                out_shot_weight = self.decoder.forward_all(out_shot_weight_s, out_shot_weight_t, act=self.weight_act)

                if self.args.weight_coef != 1:
                    out_shot_link_s = self.lin_link_s(lstm_hn[batch]).view(self.args.nb_nodes, -1)
                    out_shot_link_t = self.lin_link_t(lstm_hn[batch]).view(self.args.nb_nodes, -1)
                    out_shot_link = self.decoder.forward_all(out_shot_link_s, out_shot_link_t, act=self.link_act)
                else:
                    out_shot_link_s = None
                    out_shot_link_t = None
                    out_shot_link = out_shot_weight
                result[batch]['out_shot_link'] = out_shot_link

                if self.args.weight_coef != 1:
                    result[batch]['link_pred'] = torch.stack([self.decoder.forward_all(self.lin_link_s(lstm_h_seq[batch, x, :]).view(self.args.nb_nodes, -1),
                                              self.lin_link_t(lstm_h_seq[batch, x, :]).view(self.args.nb_nodes, -1),
                                              act=self.link_act) for x in range(lstm_h_seq.shape[1] - 1)])
                result[batch]['out_shot_weight'] = out_shot_weight
            else:
                out_shot_link_s, out_shot_link_t, out_shot_weight_s, out_shot_weight_t = None, None, None, None

            if self.args.weight_coef != 1:
                result[batch]['out_shot_link_train_s'] = [self.lin_link_s(lstm_h_seq[batch, x, :]).view(self.args.nb_nodes, -1) for x in range(lstm_h_seq.shape[1])]
                result[batch]['out_shot_link_train_t'] = [self.lin_link_t(lstm_h_seq[batch, x, :]).view(self.args.nb_nodes, -1) for x in range(lstm_h_seq.shape[1])]
            result[batch]['out_shot_weight_train_s'] = [self.lin_weight_s(lstm_h_seq[batch, x, :]).view(self.args.nb_nodes, -1) for x in range(lstm_h_seq.shape[1])]
            result[batch]['out_shot_weight_train_t'] = [self.lin_weight_t(lstm_h_seq[batch, x, :]).view(self.args.nb_nodes, -1) for x in range(lstm_h_seq.shape[1])]

        return result

    def recon_loss(self, decode_result, train_out_shots):
        batch_size = len(decode_result)

        loss = 0.0
        for batch in range(batch_size):
            edge_index_list, edge_weight_list = train_out_shots[batch][0], train_out_shots[batch][1]
            batch_result = decode_result[batch]
            node_num = batch_result['out_shot_weight_train_s'][0].size(0)
            assert len(batch_result['out_shot_weight_train_s']) == len(batch_result['out_shot_weight_train_t']) == len(edge_index_list) == len(edge_weight_list)
            for seq in range(len(batch_result['out_shot_weight_train_s'])):
                pos_edge_index, pos_edge_weight = edge_index_list[seq], edge_weight_list[seq]

                # Do not include self-loops in negative samples
                pos_edge_index_, _ = remove_self_loops(pos_edge_index)
                pos_edge_index_, _ = add_self_loops(pos_edge_index_)
                neg_edge_index = negative_sampling(pos_edge_index_, node_num)
                pos_decode_link, neg_decode_link = None, None
                if self.args.weight_coef != 1:
                    # link reconstruction loss
                    s, t = batch_result['out_shot_link_train_s'][seq], batch_result['out_shot_link_train_t'][seq]

                    pos_decode_link = self.decoder(s, t, pos_edge_index, act=self.link_act)
                    neg_decode_link = self.decoder(s, t, neg_edge_index, act=self.link_act)
                    pos_loss = -torch.log(pos_decode_link + EPS).mean()
                    neg_loss = -torch.log(1 - neg_decode_link + EPS).mean()

                    loss += (1 - self.args.weight_coef) * (pos_loss + neg_loss)
                if self.args.neg_sampling:
                    if self.args.weight_coef != 0:
                        # weight reconstruction loss
                        s, t = batch_result['out_shot_weight_train_s'][seq], batch_result['out_shot_weight_train_t'][seq]
                        pos_decode = self.decoder(s, t, pos_edge_index, act=self.weight_act)
                        neg_decode = self.decoder(s, t, neg_edge_index, act=self.weight_act)

                        pos_loss = self.loss_func(pos_decode.unsqueeze(0), pos_edge_weight.unsqueeze(0))
                        neg_loss = self.loss_func(neg_decode.unsqueeze(0), torch.zeros(neg_edge_index.size(-1)).unsqueeze(0).to(
                            pos_edge_index.device))
                        loss += self.args.weight_coef * (pos_loss + neg_loss)
                else:
                    if self.args.weight_coef != 0:
                        # weight reconstruction loss
                        s, t = batch_result['out_shot_weight_train_s'][seq], batch_result['out_shot_weight_train_t'][
                            seq]
                        source = self.decoder.forward_all(s, t, act=self.weight_act)
                        target = torch.zeros_like(source).to(s.device)
                        target[pos_edge_index[0], pos_edge_index[1]] = pos_edge_weight
                        loss += self.loss_func(source, target)
        return loss

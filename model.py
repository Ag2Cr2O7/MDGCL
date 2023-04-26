import torch
import torch.nn as nn
from utils import edge_drop, spmm
import scipy.sparse as sp
from manifold.hyperboloid import Hyperboloid
import numpy as np
from utils import scipy_sparse_mat_to_torch_sparse_tensor


class MDGCL(nn.Module):
    def __init__(self, n_u, n_i, d, train_csr, adj_norm, l, temp, Lhyper, dropout, batch_user, device, args):
        super(MDGCL, self).__init__()
        self.args = args
        self.device = device
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))
        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.temp = temp 
        self.dropout = dropout 
        self.act = nn.LeakyReLU(0.5) 
        self.batch_user = batch_user 
        self.Ws = nn.ModuleList([W_contrastive(d) for i in range(l)])
        # self.Ws1 = W_contrastive(d)
        self.E_u = None
        self.E_i = None
        self.S_u=None
        self.S_i=None
        self.c = 1
        self.Lhyper = Lhyper
        self.zero=torch.tensor([0]).to(device)
        self.manifold = Hyperboloid()
        self.fill_zero_u = torch.zeros(self.E_u_0.shape[0], 1)
        self.fill_zero_i = torch.zeros(self.E_i_0.shape[0], 1)
        self.E_u_00 = self.manifold.logmap0(self.manifold.expmap0(torch.cat((self.fill_zero_u, self.E_u_0), dim=1), self.c), self.c)
        self.E_i_00 = self.manifold.logmap0(self.manifold.expmap0(torch.cat((self.fill_zero_i, self.E_i_0), dim=1), self.c), self.c)
        self.S_u_0 = nn.parameter.Parameter(self.E_u_00)
        self.S_i_0 = nn.parameter.Parameter(self.E_i_00)
        self.S_u_list = [None] * (l + 1)
        self.S_i_list = [None] * (l + 1)
        self.S_zu_list = [None] * (l + 1)
        self.S_zi_list = [None] * (l + 1)
        self.S_u = None
        self.S_i = None
        self.S_u_list[0] = self.S_u_0
        self.S_i_list[0] = self.S_i_0
        self.E_uu_list, self.E_ii_list = [None] * (l + 1), [None] * (l + 1)
        self.E_uu_list[0], self.E_ii_list[0] = self.E_u_0, self.E_i_0
        self.Z_uu_list, self.Z_ii_list = [None] * (l + 1), [None] * (l + 1)
        self.E_uu, self.E_ii = None, None


    def forward(self, uids, iids, pos, neg, test=False):
        device=self.device
        # test
        if test==True:
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray() # (batch,items)
            mask = torch.Tensor(mask).to(device)
            preds = preds * (1-mask)
            predictions = preds.argsort(descending=True)
            return predictions
        #train
        else:
            self.GraphLayer(self.l)
            loss_r=self.Lmain(uids, pos, neg)
            loss_graph= self.graph_loss(uids, iids)
            loss_hyper=self.hyper_loss(uids, iids)
            loss = loss_r + self.Lhyper * loss_hyper + self.args.Lgraph*loss_graph
            return loss, loss_r, loss_hyper, loss_graph

    def GraphLayer(self, l):
        device=self.device
        for layer in range(1, l + 1):  # LightGCN消息传播
            zu = spmm(edge_drop(self.adj_norm, self.dropout), self.E_i_list[layer - 1], self.device)
            zi = spmm(edge_drop(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1], self.device)
            self.Z_u_list[layer] = self.act(zu)
            self.Z_i_list[layer] = self.act(zi)
            if layer>=2:
                for j in range(1,layer):
                    self.Z_u_list[layer] += self.Z_u_list[j]  # (users,dim)
                    self.Z_i_list[layer] += self.Z_i_list[j]  # (items,dim)
            zus = spmm(self.adj_norm, self.S_i_list[layer - 1], self.device)
            zis = spmm(self.adj_norm.transpose(0, 1), self.S_u_list[layer - 1], self.device)
            self.S_zu_list[layer] = self.act(zus)
            self.S_zi_list[layer] = self.act(zis)
            if layer >= 2:
                for j in range(1, layer):
                    self.S_zu_list[layer] += self.S_zu_list[j]  # (29601,32)
                    self.S_zi_list[layer] += self.S_zi_list[j]  # (24734,32)
            zuu = spmm(edge_drop(self.adj_norm, self.dropout), self.E_ii_list[layer - 1], self.device)
            zii = spmm(edge_drop(self.adj_norm, self.dropout).transpose(0, 1), self.E_uu_list[layer - 1],
                       self.device)
            self.Z_uu_list[layer] = self.act(zuu)
            self.Z_ii_list[layer] = self.act(zii)
            if layer >= 2:
                for j in range(1, layer):
                    self.Z_uu_list[layer] += self.Z_uu_list[j]  # (users,dim)
                    self.Z_ii_list[layer] += self.Z_ii_list[j]  # (items,dim)
            self.E_u_list[layer] = self.Z_u_list[layer] + self.E_u_list[layer - 1]
            self.E_i_list[layer] = self.Z_i_list[layer] + self.E_i_list[layer - 1]
            self.S_u_list[layer] = self.S_zu_list[layer] + self.S_u_list[layer - 1]
            self.S_i_list[layer] = self.S_zi_list[layer] + self.S_i_list[layer - 1]
            self.E_uu_list[layer] = self.Z_uu_list[layer] + self.E_uu_list[layer - 1]
            self.E_ii_list[layer] = self.Z_ii_list[layer] + self.E_ii_list[layer - 1]
        self.E_u = sum(self.E_u_list)  # (users,dim)
        self.E_i = sum(self.E_i_list)

    def inner_product(self,a, b):
        return torch.sum(a * b, dim=-1)

    def Lss(self,uembs1,uembs2,iembs1,iembs2,t):
        tot_ratings_user = torch.matmul(uembs1,torch.transpose(uembs2, 0, 1))
        pos_ratings_user = self.inner_product(uembs1, uembs2)
        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]
        loss_u = torch.logsumexp(ssl_logits_user / t, dim=1)
        tot_ratings_item = torch.matmul(iembs1, torch.transpose(iembs2, 0, 1))
        pos_ratings_item = self.inner_product(iembs1, iembs2)
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]
        loss_i = torch.logsumexp(ssl_logits_item / t, dim=1)
        loss=torch.sum(loss_u)+torch.sum(loss_i)
        return loss


    def Lmain(self, uids, pos, neg):
        device = self.device
        loss_r = 0
        for i in range(len(uids)):
            u = uids[i]  # batch user id
            u_emb = self.E_u[u]
            u_pos = pos[i]
            u_neg = neg[i]
            pos_emb = self.E_i[u_pos].to(device)
            neg_emb = self.E_i[u_neg].to(device)
            pos_scores = u_emb @ pos_emb.T
            neg_scores = u_emb @ neg_emb.T
            hinge = nn.functional.relu(1 - pos_scores + neg_scores)
            loss_r = loss_r + hinge.sum()
        loss_r = loss_r / self.batch_user
        return loss_r


    def hyper_loss(self, uids, iids):
        device = self.device
        loss_s = 0
        for l in range(1, self.l + 1):
            # user
            gnn_u=self.Z_u_list[l][uids]
            fill_u = torch.zeros(gnn_u.shape[0], 1).to(device)
            gnn_u=torch.cat((fill_u,gnn_u),dim=1)
            gnn_u = nn.functional.normalize(gnn_u, p=2, dim=1)
            hyper_u=self.manifold.expmap0(self.S_zu_list[l][uids],self.c)
            hyper_u = nn.functional.normalize(hyper_u, p=2, dim=1)
            hyper_u = self.Ws[l - 1](hyper_u)
            pos_score = torch.exp((gnn_u * hyper_u).sum(1) / self.temp)
            neg_score = torch.exp(gnn_u @ hyper_u.T / self.temp).sum(1)
            loss_s_u = (-1 * torch.log(pos_score / (neg_score + 1e-8) + 1e-8)).sum()
            loss_s = loss_s + loss_s_u
            # item
            gnn_i=self.Z_i_list[l][iids]
            fill_i = torch.zeros(gnn_i.shape[0], 1).to(device)
            gnn_i = torch.cat((fill_i, gnn_i), dim=1)
            gnn_i = nn.functional.normalize(gnn_i, p=2, dim=1)
            hyper_i=self.manifold.expmap0(self.S_zi_list[l][iids],self.c)
            hyper_i = nn.functional.normalize(hyper_i, p=2, dim=1)
            hyper_i = self.Ws[l - 1](hyper_i)
            pos_score = torch.exp((gnn_i * hyper_i).sum(1) / self.temp)
            neg_score = torch.exp(gnn_i @ hyper_i.T / self.temp).sum(1)
            loss_s_i = (-1 * torch.log(pos_score / (neg_score + 1e-8) + 1e-8)).sum()
            # loss
            loss_s = loss_s + loss_s_i
        return loss_s


    def graph_loss(self, uids, iids):
        loss_zq = 0
        for l in range(1, self.l + 1):
            # user
            gnn_u = nn.functional.normalize(self.Z_u_list[l][uids], p=2, dim=1)
            gnn_uu = nn.functional.normalize(self.Z_uu_list[l][uids], p=2, dim=1)
            # item
            gnn_i = nn.functional.normalize(self.Z_i_list[l][iids], p=2, dim=1)
            gnn_ii = nn.functional.normalize(self.Z_ii_list[l][iids], p=2, dim=1)
            loss_zq = loss_zq + self.Lss(gnn_u, gnn_uu, gnn_i, gnn_ii,self.args.t)
        return loss_zq

class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        #self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d,d)))
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d+1, d+1)))

    def forward(self,x):
        return x @ self.W

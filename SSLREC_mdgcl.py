import torch as t
import numpy as np
import torch.cuda
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from manifold.hyperboloid import Hyperboloid
from models.model_utils import SpAdjEdgeDrop
from models.loss_utils import cal_bpr_loss, reg_params


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
class MDGCL(BaseModel):
    def __init__(self, data_handler):
        super(MDGCL, self).__init__(data_handler)
        train_mat = data_handler._load_one_mat(data_handler.trn_file)
        rowD = np.array(train_mat.sum(1)).squeeze()
        colD = np.array(train_mat.sum(0)).squeeze()
        for i in range(len(train_mat.data)):
            train_mat.data[i] = train_mat.data[i] / pow(rowD[train_mat.row[i]] * colD[train_mat.col[i]], 0.5)
        adj_norm = self._scipy_sparse_mat_to_torch_sparse_tensor(train_mat)
        self.adj_norm = adj_norm.coalesce().cuda()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.reg_weight = configs['model']['reg_weight']
        self.l = configs['model']['layer_num']
        self.lhyper = configs['model']['hyper']
        self.lgraph = configs['model']['graph']
        self.dropout = configs['model']['dropout']
        self.temp = configs['model']['temp']
        self.t = configs['model']['t']
        self.E_u_0 = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.E_i_0 = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.E_u_list = [None] * (self.l+1) # [None, None, None]
        self.E_i_list = [None] * (self.l+1) # [None, None, None]
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (self.l+1)
        self.Z_i_list = [None] * (self.l+1)
        self.act = nn.LeakyReLU(0.5) 
        self.Ws = nn.ModuleList([W_contrastive(self.embedding_size) for i in range(self.l)])
        self.E_u = None
        self.E_i = None
        self.S_u=None
        self.S_i=None
        self.c = 1
        self.zero=torch.tensor([0]).to(self.device)
        self.manifold = Hyperboloid()
        self.fill_zero_u = torch.zeros(self.E_u_0.shape[0], 1)
        self.fill_zero_i = torch.zeros(self.E_i_0.shape[0], 1)
        self.E_u_00 = self.manifold.logmap0(self.manifold.expmap0(torch.cat((self.fill_zero_u, self.E_u_0), dim=1), self.c), self.c)
        self.E_i_00 = self.manifold.logmap0(self.manifold.expmap0(torch.cat((self.fill_zero_i, self.E_i_0), dim=1), self.c), self.c)
        self.S_u_0 = nn.parameter.Parameter(self.E_u_00)
        self.S_i_0 = nn.parameter.Parameter(self.E_i_00)
        self.S_u_list = [None] * (self.l + 1)
        self.S_i_list = [None] * (self.l + 1)
        self.S_zu_list = [None] * (self.l + 1)
        self.S_zi_list = [None] * (self.l + 1)
        self.S_u = None
        self.S_i = None
        self.S_u_list[0] = self.S_u_0
        self.S_i_list[0] = self.S_i_0
        self.E_uu_list, self.E_ii_list = [None] * (self.l + 1), [None] * (self.l + 1)  # [None, None, None]
        self.E_uu_list[0], self.E_ii_list[0] = self.E_u_0, self.E_i_0
        self.Z_uu_list, self.Z_ii_list = [None] * (self.l + 1), [None] * (self.l + 1)
        self.E_uu, self.E_ii = None, None
        self.is_training = True
        self.zero = torch.tensor([0]).cuda()

    def _scipy_sparse_mat_to_torch_sparse_tensor(self,sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data)
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)

    def _spmm(self,sp, emb):
        sp = sp.coalesce()
        cols = sp.indices()[1]
        rows = sp.indices()[0]
        col_segs =  emb[cols] * t.unsqueeze(sp.values(),dim=1)
        result = t.zeros((sp.shape[0],emb.shape[1])).cuda()
        result.index_add_(0, rows, col_segs)
        return result

    def _sparse_dropout(self,mat, dropout):
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()
        return t.sparse.FloatTensor(indices, values, size)

    def forward(self,test=False):
        if test and self.E_u is not None:
            return self.E_u, self.E_i
        else:  # training phase
            for layer in range(1, self.l + 1):  
                zu = self._spmm(self._sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1])
                zi = self._spmm(self._sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1])
                #self.Z_u_list[layer] = self.act(zu)  # (29601,32)
                #self.Z_i_list[layer] = self.act(zi)  # (24734,32)
                self.Z_u_list[layer] = zu  # (29601,32)
                self.Z_i_list[layer] = zi  # (24734,32)
                if layer >= 2:
                    for j in range(1, layer):
                        self.Z_u_list[layer] += self.Z_u_list[j]  # (29601,32)
                        self.Z_i_list[layer] += self.Z_i_list[j]  # (24734,32)
                zus = self._spmm(self.adj_norm, self.S_i_list[layer - 1])
                zis = self._spmm(self.adj_norm.transpose(0, 1), self.S_u_list[layer - 1])
                # self.S_zu_list[layer] = self.act(zus)
                # self.S_zi_list[layer] = self.act(zis)
                self.S_zu_list[layer] = zus
                self.S_zi_list[layer] = zis
                if layer >= 2:
                    for j in range(1, layer):
                        self.S_zu_list[layer] += self.S_zu_list[j]  # (29601,32)
                        self.S_zi_list[layer] += self.S_zi_list[j]  # (24734,32)
                zuu = self._spmm(self._sparse_dropout(self.adj_norm, self.dropout), self.E_ii_list[layer-1])
                zii = self._spmm(self._sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_uu_list[layer-1])
                # self.Z_uu_list[layer] = self.act(zuu)
                # self.Z_ii_list[layer] = self.act(zii)
                self.Z_uu_list[layer] =zuu
                self.Z_ii_list[layer] = zii
                if layer >= 2:
                    for j in range(1, layer):
                        self.Z_uu_list[layer] += self.Z_uu_list[j]  # (29601,32)
                        self.Z_ii_list[layer] += self.Z_ii_list[j]  # (24734,32)

                self.E_u_list[layer] = self.Z_u_list[layer] + self.E_u_list[layer - 1]
                self.E_i_list[layer] = self.Z_i_list[layer] + self.E_i_list[layer - 1]

                self.S_u_list[layer] = self.S_zu_list[layer] + self.S_u_list[layer - 1]
                self.S_i_list[layer] = self.S_zi_list[layer] + self.S_i_list[layer - 1]

                self.E_uu_list[layer] = self.Z_uu_list[layer] + self.E_uu_list[layer - 1]
                self.E_ii_list[layer] = self.Z_ii_list[layer] + self.E_ii_list[layer - 1]
            self.E_u = sum(self.E_u_list)  # (users,32)
            self.E_i = sum(self.E_i_list)
            return self.E_u, self.E_i


    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data #ancs为uids
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        uids=ancs
        iids = t.cat((poss,negs))
        hyper_loss=self.lhyper*self.hyper_loss(uids,iids)
        #hyper_loss=self.zero
        graph_loss=self.lgraph*self.graph_loss(uids,iids)
        reg_loss = reg_params(self) * self.reg_weight
        loss = bpr_loss + hyper_loss+graph_loss+reg_loss
        losses = {'bpr_loss': bpr_loss, 'hyper_loss': hyper_loss,'graph_loss': graph_loss}  
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(test=True)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

    def inner_product(self,a, b):
        return torch.sum(a * b, dim=-1) 

    def Lss(self,uembs1,uembs2,iembs1,iembs2,t):
        device=self.device
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



    def hyper_loss(self,uids,iids):
        device = self.device
        loss_s = 0
        for l in range(1, self.l + 1):
            gnn_u=self.Z_u_list[l][uids]
            fill_u = torch.zeros(gnn_u.shape[0], 1).to(device)
            gnn_u=torch.cat((fill_u,gnn_u),dim=1)
            gnn_u = nn.functional.normalize(gnn_u, p=2, dim=1)
            hyper_u=self.manifold.expmap0(self.S_zu_list[l][uids],self.c)
            #hyper_u=self.manifold.proj(self.manifold.expmap0(self.S_zu_list[l][uids],self.c),self.c)
            hyper_u = nn.functional.normalize(hyper_u, p=2, dim=1)
            hyper_u = self.Ws[l - 1](hyper_u)  # hyper_u*(32,32)
            pos_score = torch.exp((gnn_u * hyper_u).sum(1) / self.temp)  # (256,32)->(256) 
            neg_score = torch.exp(gnn_u @ hyper_u.T / self.temp).sum(1)  # (256*256)->(256)
            loss_s_u = (-1 * torch.log(pos_score / (neg_score + 1e-8) + 1e-8)).sum()
            loss_s = loss_s + loss_s_u  
            # Item loss
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
            loss_s = loss_s + loss_s_i
        return loss_s



    def graph_loss(self,uids,iids):
        loss_zq = 0
        for l in range(1, self.l + 1):
            gnn_u = nn.functional.normalize(self.Z_u_list[l][uids], p=2, dim=1)  # (256,32)
            gnn_uu = nn.functional.normalize(self.Z_uu_list[l][uids], p=2, dim=1)
         
            gnn_i = nn.functional.normalize(self.Z_i_list[l][iids], p=2, dim=1)
            gnn_ii = nn.functional.normalize(self.Z_ii_list[l][iids], p=2, dim=1)
            loss_zq = loss_zq + self.Lss(gnn_u, gnn_uu, gnn_i, gnn_ii,self.t) 
        return loss_zq

class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        #self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d,d)))
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d+1, d+1)))

    def forward(self,x):
        return x @ self.W 

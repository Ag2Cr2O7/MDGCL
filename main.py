import os
import numpy as np
import torch
import pickle
from model import MDGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor,set_seed,convert_number,learning_rate_decay
import pandas as pd
from myparser import parse_args
from tqdm import tqdm
from time import time
from log import Logger,model_info
import logging
from datetime import datetime

dataset= 'yelp'
args=parse_args(dataset=dataset)
#args.use_log=False
seed=512
graph=convert_number(args.Lgraph)
t=args.t
hyper=convert_number(args.Lhyper)
temp=args.temp
lreg=convert_number(args.lambda2)
nowtime=str(datetime.now())[:-7]
runname= str(seed) + 'graph' + graph + 't' + str(t) + 'hyper' + hyper + 't\'' + str(temp) + 'l2reg' + lreg + '_' + nowtime[-2:]
print(nowtime)
print('runname:',runname)


devicestr = 'use GPU' if torch.cuda.is_available() else 'use CPU'
print(devicestr)
device = 'cuda:' + args.cuda if torch.cuda.is_available() else 'cpu'
use_gpu=True if torch.cuda.is_available() else False

set_seed(seed)
# hyperparameters
d = args.d
l = args.gnn_layer
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
l_hyper = args.Lhyper
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr

# load data
path = 'data/' + args.data + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)

train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
print('Data '+'./data/' + args.data + '/'+' loaded.')
epoch_user = min(train.shape[0], 30000) #max 30000
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)

rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
print('Calculate Nu and Ni...')
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().to(device)
print('Adj matrix normalized.')

test_labels = [[] for i in range(test.shape[0])]
print('Generate test data...')
for i in range(len(test.data)):
    row = test.row[i] # user id
    col = test.col[i] # item id
    test_labels[row].append(col)
print('Test data processed.')

# log
if args.use_log==True:
    log_path = 'save_log/' + dataset + '/' + runname + '.log'
    print('Directory of log :', log_path)
    log = Logger(log_path)
    model_info(log, args, runname)
    log.info('-' * 48 + 'train' + '-' * 48)
    log.set_log()
    log.info('Model Train')


print('Users:', train.shape[0],'Items:', train.shape[1],'Interactions:',len(train.data))
print('embedding dim:', args.d,'Lgraph:', graph, 't:', args.t, 'Lhyper:', hyper,'t\':', temp ,'l2reg:', convert_number(lambda_2))

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []
best_recall1,best_ndcg1,best_recall2,best_ndcg2,best_epoch=0,0,0,0,0
early_stop_count = 0
early_stop = False

model = MDGCL(adj_norm.shape[0], adj_norm.shape[1], d, train_csr, adj_norm, l, temp, l_hyper, dropout, batch_user, device, args)
#model.load_state_dict(torch.load('saved_model.pt'))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),weight_decay=lambda_2,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

current_lr = lr

#train
for epoch in range(epoch_no):
    current_lr = learning_rate_decay(optimizer)
    e_users = np.random.permutation(adj_norm.shape[0])[:epoch_user]
    batch_no = int(np.ceil(epoch_user/batch_user))
    iter_train_batch = tqdm(range(batch_no))
    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    epoch_loss_n =0
    for batch in iter_train_batch:
        start = batch*batch_user
        end = min((batch+1)*batch_user,epoch_user)
        batch_users = e_users[start:end]
        # sample pos and neg
        pos = []
        neg = []
        iids = set()
        for i in range(len(batch_users)): #256
            u = batch_users[i]
            u_interact = train_csr[u].toarray()[0]
            positive_items = np.random.permutation(np.where(u_interact==1)[0])
            negative_items = np.random.permutation(np.where(u_interact==0)[0])
            item_num = min(max_samp,len(positive_items))
            positive_items = positive_items[:item_num]
            negative_items = negative_items[:item_num]
            # pos,neg
            pos.append(torch.LongTensor(positive_items).to(device))
            neg.append(torch.LongTensor(negative_items).to(device))
            iids = iids.union(set(positive_items))
            iids = iids.union(set(negative_items))
        # end for
        iids = torch.LongTensor(list(iids)).to(device)
        uids = torch.LongTensor(batch_users).to(device)
        ####
        optimizer.zero_grad()
        loss, loss_r, loss_s,loss_cse= model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()     # Loss
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()
        epoch_loss_n+=loss_cse.cpu().item()
        torch.cuda.empty_cache()

    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    epoch_loss_n =epoch_loss_n/batch_no
    # save epoch loss
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    print('Epoch:{} Loss{:.4f} Lrec:{:.4f} LGraph:{:.4f} LHyper:{:.4f} '.format(epoch,epoch_loss,epoch_loss_r,epoch_loss_n,epoch_loss_s))
    # generate log
    strepoch=str(epoch)
    strepoch=' '+strepoch if len(strepoch)<2 else strepoch
    loss_info='Epoch:'+strepoch+' Loss:'+str(epoch_loss)[:7]+' Lrec:'+str(epoch_loss_r)[:5]+' LGraph:'+str(epoch_loss_n)[:7]+' LHyper:'+str(epoch_loss_s)[:7]
    if args.use_log == True:
        log.info(loss_info)

    # test freq
    if epoch % 3 == 0:
        test_uids = np.array([i for i in range(adj_norm.shape[0])]) # users
        batch_no = int(np.ceil(len(test_uids)/batch_user))
        # ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
        iter_test_batch = tqdm(range(batch_no))
        all_recall_20 = 0
        all_ndcg_20 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        for batch in iter_test_batch:
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))
            test_uids_input = torch.LongTensor(test_uids[start:end]).to(device)
            predictions = model(test_uids_input,None,None,None,test=True)
            predictions = np.array(predictions.cpu())
            recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
            #top@40
            recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)
            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_recall_40+=recall_40
            all_ndcg_40+=ndcg_40
            #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
        print('----------------------------------------------------')
        print('Test of epoch',epoch,':','Recall@20:',all_recall_20/batch_no,'NDCG@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'NDCG@40:',all_ndcg_40/batch_no)
        score_info='Test Epoch:'+strepoch+' Recall@20:'+str(all_recall_20/batch_no)[:10]+' NDCG@20:'+str(all_ndcg_20/batch_no)[:10]+' Recall@40:'+str(all_recall_40/batch_no)[:10]+' NDCG@40:'+str(all_ndcg_40/batch_no)[:10]
        if args.use_log == True:
            log.set_log(format=logging.Formatter())
            log.info(score_info)
            log.set_log(format=logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))  
        recall_20_x.append(epoch)
        recall_20_y.append(all_recall_20/batch_no)
        ndcg_20_y.append(all_ndcg_20/batch_no)
        recall_40_y.append(all_recall_40/batch_no)
        ndcg_40_y.append(all_ndcg_40/batch_no)
        # best
        if all_recall_20/batch_no > best_recall1:
            best_recall1, best_ndcg1, best_epoch = all_recall_20/batch_no, all_ndcg_20/batch_no, epoch
            best_recall2, best_ndcg2=all_recall_40/batch_no,all_ndcg_40/batch_no
            early_stop_count = 0
            # bestmodel=model.state_dict()
            # best_opti=optimizer.state_dict()
        else:
            early_stop_count += 1
            if early_stop_count == args.earlystop:
                early_stop = True
    if early_stop:
        print('-----------------------------------------------------')
        print('Early stop is triggered at {} epochs.'.format(epoch))
        if args.use_log == True:
            log.set_log(format=logging.Formatter()) 
            log.info('-' * 48+'result'+'-'*48)
            log.info('Early stop is triggered at '+str(epoch)+ ' epochs.')
        break
print('-----------------'+str(datetime.now())[:-7]+'-------------------')
#print(str(datetime.now())[:-7])
print('Best:'+str(best_epoch))
print('Recall@20:',best_recall1,'Ndcg@20:',best_ndcg1, 'Recall@40:',best_recall2, 'Ndcg@40:',best_ndcg2)
#保存最佳的数据
recall_20_x.append('Best:'+str(best_epoch))
recall_20_y.append(best_recall1)
ndcg_20_y.append(best_ndcg1)
recall_40_y.append(best_recall2)
ndcg_40_y.append(best_ndcg2)
if args.use_log==True:
    if not early_stop:
        log.set_log(format=logging.Formatter())
        log.info('-' * 48+'result'+'-'*47)
    log.set_log(format=logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    log.info('Best Recall & NDCG')
    log.set_log(format=logging.Formatter())
    log.info('Best epoch is '+str(best_epoch))
    log.info('Recall@20:'+str(best_recall1)+' Ndcg@20:'+str(best_ndcg1)+' Recall@40:'+str(best_recall2)+' Ndcg@40:'+str(best_ndcg2))
    print('Log file:',log_path)
print('Rec metrics:','log/' + dataset + runname + '.csv')

metric = pd.DataFrame({
    'epoch':recall_20_x,
    'recall@20':recall_20_y,
    'ndcg@20':ndcg_20_y,
    'recall@40':recall_40_y,
    'ndcg@40':ndcg_40_y
})
metric.to_csv('log_csv/' + dataset + runname + '.csv')

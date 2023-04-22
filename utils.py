import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import random
import pickle
from tqdm import trange
import pandas as pd
from scipy.sparse import coo_matrix,csr_matrix

def learning_rate_decay(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']*0.98
        if lr > 0.0005:
            param_group['lr'] = lr
    return lr


def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i] #user id
        prediction = list(predictions[i][:topk]) # top-K item id
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    if user_num==0: return 0,0
    return all_recall/user_num, all_ndcg/user_num

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def edge_drop(mat, dropout):
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)


def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device)) if device!='cpu' else torch.zeros((sp.shape[0], emb.shape[1]))
    result.index_add_(0, rows, col_segs)
    return result


def set_seed(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def pkl2txt(path='',seg='\t'):
     print('convert'+path)
     f1 = open(path + 'trnMat.pkl', 'rb')
     train = pickle.load(f1)  # (u1,i5)->1.0 (u1,i8)->1.0
     a = train.row
     b = train.col
     train=[]
     fp1=open(path+'train.txt','w',encoding='utf-8')
     for i in tqdm(range(len(a))):
        train.append([a[i],b[i]])
     train.sort(key=lambda x: x[1])
     train.sort(key=lambda x: x[0])
     for i in tqdm(range(len(train))):
        fp1.write(str(train[i][0])+seg+str(train[i][1]))
        fp1.write('\n')
     f1.close()
     fp1.close()
     f2 = open(path + 'tstMat.pkl', 'rb')
     test = pickle.load(f2)  # (u1,i5)->1.0 (u1,i8)->1.0
     c = test.row
     d = test.col
     test=[]
     fp2 = open(path+'test.txt', 'w', encoding='utf-8')
     for i in tqdm(range(len(c))):
       test.append([c[i],d[i]])
     test.sort(key=lambda x: x[1])
     test.sort(key=lambda x: x[0])
     for i in tqdm(range(len(test))):
         fp2.write(str(test[i][0])+seg+str(test[i][1]))
         fp2.write('\n')
     f2.close()
     fp2.close()
     print('ok！')

def coltxt2pkl(path,sep='\t'):
    #train
    traindata = pd.read_table(path + 'train.txt', header=None, sep=sep)
    train_user = traindata.values[:, 0]  # 第0列
    train_item = traindata.values[:, 1]
    row = np.array(train_user)
    col = np.array(train_item)
    flag = np.array(np.ones(len(train_user)))
    coo = coo_matrix((flag, (row, col)))
    print(coo.shape)
    fp = open(path+'trnMat.pkl', 'wb')
    pickle.dump(coo, fp)
    print(path + 'trnMat.pkl')
    #test
    testdata = pd.read_table(path + 'test.txt', header=None, sep=sep)
    test_user = testdata.values[:, 0]
    test_item = testdata.values[:, 1]
    row1 = np.array(test_user)
    col1 = np.array(test_item)
    flag1 = np.array(np.ones(len(test_user)))
    coo1 = coo_matrix((flag1, (row1, col1)))
    print(coo1.shape)
    fp1 = open(path+'tstMat.pkl', 'wb')
    pickle.dump(coo1, fp1)
    print(path+'tstMat.pkl')


def convert_number(x):
    y=list("%.0e"%x)
    del y[-2]
    y="".join(y)
    return y


# if __name__=='__main__':
#     # numpy==1.17.0
#     path = 'data/' + 'tmall' + '/'
#     seg=' '
#     coltxt2pkl(path,seg)
#     #pkl2txt(path,seg)


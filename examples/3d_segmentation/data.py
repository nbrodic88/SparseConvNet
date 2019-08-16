# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch, torch.utils.data
import glob, math, os
import scipy, scipy.ndimage
import sparseconvnet as scn

if not os.path.exists('train_val/'):
    print('Downloading data ...')
    os.system('bash download_data.sh')

categories=["000001", "000002", "000003", "000004",
       "000005", "000006"]
classes=['Bridge', 'Building',      'Ground',        'Trees',
         'Unlabeled',    'Water']
nClasses=[6, 6, 6, 6, 6, 6]
classOffsets=np.cumsum([0]+nClasses)

def init(c,resolution=50,sz=50*8+8,batchSize=16):
    globals()['categ']=c
    globals()['resolution']=resolution
    globals()['batchSize']=batchSize
    globals()['spatialSize']=torch.LongTensor([sz]*3)
    if categ==-1:
        print('Total 6 classes')
        globals()['nClassesTotal']=int(classOffsets[-1])
    else:
        print('categ ',categ,classes[categ])
        globals()['nClassesTotal']=int(nClasses[categ])

def load(xF, c, classOffset, nc):
    xl=np.loadtxt(xF[0])
    xl/= ((xl**2).sum(1).max()**0.5)
    y = np.loadtxt(xF[0][:-9]+'txt').astype('int64')+classOffset-1
    return (xF[0], xl, y, c, classOffset, nc, np.random.randint(1e6))

def train():
    d=[]
    if categ==-1:
        for c in range(6):
            for x in torch.utils.data.DataLoader(
                glob.glob('train_val/'+categories[c]+'/*.txt.train'),
                collate_fn=lambda x: load(x, c, classOffsets[c],nClasses[c]),
                num_workers=12):
                d.append(x)
    else:
        for x in torch.utils.data.DataLoader(
            glob.glob('train_val/'+categories[categ]+'/*.txt.train'),
            collate_fn=lambda x: load(x, categ, 0, nClasses[categ]),
            num_workers=12):
            d.append(x)

    print(len(d))
    def merge(tbl):
        xl_=[]
        xf_=[]
        y_=[]
        categ_=[]
        mask_=[]
        classOffset_=[]
        nClasses_=[]
        nPoints_=[]
        np_random=np.random.RandomState([x[-1] for x in tbl])
        for _, xl, y, categ, classOffset, nClasses, idx in tbl:
            m=np.eye(3,dtype='float32')
            m[0,0]*=np_random.randint(0,2)*2-1
            m=np.dot(m,np.linalg.qr(np_random.randn(3,3))[0])
            xl=np.dot(xl,m)
            xl+=np_random.uniform(-1,1,(1,3)).astype('float32')
            xl=np.floor(resolution*(4+xl)).astype('int64')
            xf=np.ones((xl.shape[0],1)).astype('float32')
            xl_.append(xl)
            xf_.append(xf)
            y_.append(y)
            categ_.append(np.ones(y.shape[0],dtype='int64')*categ)
            classOffset_.append(classOffset)
            nClasses_.append(nClasses)
            mask=np.zeros((y.shape[0],nClassesTotal),dtype='float32')
            mask[:,classOffset:classOffset+nClasses]=1
            mask_.append(mask)
            nPoints_.append(y.shape[0])
        xl_=[np.hstack([x,idx*np.ones((x.shape[0],1),dtype='int64')]) for idx,x in enumerate(xl_)]
        return {'x':  [torch.from_numpy(np.vstack(xl_)),torch.from_numpy(np.vstack(xf_))],
                'y':           torch.from_numpy(np.hstack(y_)),
                'categ':       torch.from_numpy(np.hstack(categ_)),
                'classOffset': classOffset_,
                'nClasses':    nClasses_,
                'mask':        torch.from_numpy(np.vstack(mask_)),
                'xf':          [x[0] for x in tbl],
                'nPoints':     nPoints_}
    return torch.utils.data.DataLoader(d,batch_size=batchSize, collate_fn=merge, num_workers=10, shuffle=True)

def valid():
    d=[]
    if categ==-1:
        for c in range(6):
            for x in torch.utils.data.DataLoader(
                glob.glob('train_val/'+categories[c]+'/*.txt.valid'),
                collate_fn=lambda x: load(x, c, classOffsets[c],nClasses[c]),
                num_workers=12):
                d.append(x)
    else:
        for x in torch.utils.data.DataLoader(
            glob.glob('train_val/'+categories[categ]+'/*.txt.valid'),
            collate_fn=lambda x: load(x, categ, 0, nClasses[categ]),
            num_workers=12):
            d.append(x)
    print(len(d))
    def merge(tbl):
        xl_=[]
        xf_=[]
        y_=[]
        categ_=[]
        mask_=[]
        classOffset_=[]
        nClasses_=[]
        nPoints_=[]
        np_random=np.random.RandomState([x[-1] for x in tbl])
        for _, xl, y, categ, classOffset, nClasses, idx in tbl:
            m=np.eye(3,dtype='float32')
            m[0,0]*=np_random.randint(0,2)*2-1
            m=np.dot(m,np.linalg.qr(np_random.randn(3,3))[0])
            xl=np.dot(xl,m)
            xl+=np_random.uniform(-1,1,(1,3)).astype('float32')
            xl=np.floor(resolution*(4+xl)).astype('int64')
            xl_.append(xl)
            xf=np.ones((xl.shape[0],1)).astype('float32')
            xf_.append(xf)
            y_.append(y)
            categ_.append(np.ones(y.shape[0],dtype='int64')*categ)
            classOffset_.append(classOffset)
            nClasses_.append(nClasses)
            mask=np.zeros((y.shape[0],nClassesTotal),dtype='float32')
            mask[:,classOffset:classOffset+nClasses]=1
            mask_.append(mask)
            nPoints_.append(y.shape[0])
        xl_=[np.hstack([x,idx*np.ones((x.shape[0],1),dtype='int64')]) for idx,x in enumerate(xl_)]
        return {'x':  [torch.from_numpy(np.vstack(xl_)),torch.from_numpy(np.vstack(xf_))],
                'y':           torch.from_numpy(np.hstack(y_)),
                'categ':       torch.from_numpy(np.hstack(categ_)),
                'classOffset': classOffset_,
                'nClasses':    nClasses_,
                'mask': torch.from_numpy(np.vstack(mask_)),
                'xf':          [x[0] for x in tbl],
                'nPoints':     nPoints_}
    return torch.utils.data.DataLoader(d,batch_size=batchSize, collate_fn=merge, num_workers=10, shuffle=True)

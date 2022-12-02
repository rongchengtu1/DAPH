import pickle
import os
import argparse
import logging
import torch
import time
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import h5py
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dataset_my as dp
import utils.hash_model as hash_models
import utils.calc_hr as calc_hr
import torch.nn as nn


def GenerateCode(model_hash, model_text, data_loader, num_data, bit, k=0):
    B = np.zeros((num_data, bit), dtype=np.float32)
    Bt = np.zeros((num_data, bit), dtype=np.float32)
    kk = 1
    for iter, data in enumerate(data_loader, 0):
        data_img, text, train_label,  data_ind = data
        data_img = Variable(data_img.cuda())
        train_label = torch.squeeze(train_label)
        train_label = Variable(train_label.type(torch.FloatTensor).cuda())
        text = Variable(text.type(torch.FloatTensor).cuda())
        if k == 0:
            out = model_hash(data_img, train_label)
            outt = model_text(text, train_label)
            if kk:
                # print(out.item())
                kk = 0
            B[data_ind.numpy(), :] = torch.sign(out.data.cpu()).numpy()
            Bt[data_ind.numpy(), :] = torch.sign(outt.data.cpu()).numpy()
    return B, Bt

def DAPH_proxy_code_Algo(opt):
    code_length = opt.bit
    bit = opt.bit
    data_set = opt.dataset

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    batch_size = 90
    epochs = 80
    learning_rate = 0.0002 #0.05
    weight_decay = 10 ** -5
    model_name = 'alexnet'

    gamma = opt.gamma
    sigma = opt.sigma

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_dta = h5py.File(opt.data_path)
    text_len = np.asarray(all_dta['test_y']).shape[1]
    dset_train = dp.DatasetPorcessing_h5(
        np.asarray(all_dta['train_data']), np.asarray(all_dta['train_y']), np.asarray(all_dta['train_L']),
        transformations)

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    train_labels = dset_train.labels



    label_size = train_labels.size()
    nclass = label_size[1]

    hash_text_model = hash_models.Label_text_Net(text_len, nclass, code_length)
    hash_text_model.cuda()


    optimizer_hash_text = torch.optim.SGD(hash_text_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler_text = torch.optim.lr_scheduler.StepLR(optimizer_hash_text, step_size=100, gamma=0.3, last_epoch=-1)

    hash_model = hash_models.Label_Net(model_name, nclass, code_length)
    hash_model.cuda()
    optimizer_hash = torch.optim.SGD([{'params': hash_model.features.parameters(), 'lr': learning_rate},
                                  {'params': hash_model.classifier.parameters(), 'lr': learning_rate}],
                                 lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hash, step_size=100, gamma=0.3, last_epoch=-1)

    label_model = hash_models.LabelNet(nclass, code_length)
    label_model.cuda()

    optimizer_label = optim.SGD(label_model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=100, gamma=0.1, last_epoch=-1)

    labels_cate = torch.eye(nclass).type(torch.FloatTensor).cuda()
    I = torch.eye(nclass).type(torch.FloatTensor).cuda()
    relu = nn.ReLU()
    eps=1e-5
    for epoch in range(epochs):
        label_model.train()
        hash_model.train()
        hash_text_model.train()
        scheduler_l.step()
        scheduler_text.step()
        scheduler.step()
        epoch_loss = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_img, train_text, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)
            train_img = Variable(train_img.cuda())
            train_label = Variable(train_label.type(torch.FloatTensor).cuda())
            train_text = Variable(train_text.type(torch.FloatTensor).cuda())
            the_batch = len(batch_ind)
            hash_out = hash_model(train_img, train_label)
            text_out = hash_text_model(train_text, train_label)
            label_out = label_model(labels_cate)

            loss1 = relu((label_out.mm(label_out.t()) - I)).pow(2).sum()


            logit = hash_out.mm(label_out.t())

            our_logit = torch.exp((logit - sigma) * gamma) * train_label
            mu_logit = torch.exp(logit * (1 - train_label) * gamma).sum(1).view(-1, 1).expand(the_batch, train_label.size()[1]) + our_logit
            loss = - ((torch.log(our_logit / (mu_logit + eps) + eps + 1 - train_label)).sum(1) / train_label.sum(1)).sum()

            logit_text = text_out.mm(label_out.t())

            our_logit_text = torch.exp((logit_text - sigma) * gamma) * train_label
            mu_logit_text = torch.exp(logit_text * (1 - train_label) * gamma).sum(1).view(-1, 1).expand(the_batch, train_label.size()[1]) + our_logit_text
            loss_text = - ((torch.log(our_logit_text / (mu_logit_text + eps) + eps + 1 - train_label)).sum(1) / train_label.sum(1)).sum()

            loss_b = label_out.sum(0).pow(2).sum() + text_out.sum(0).pow(2).sum() + hash_out.sum(0).pow(2).sum()

            B = F.normalize(torch.sign(hash_out))
            reg1 = ((B * hash_out).sum(1) - 1.).pow(2).sum()
            Bt = F.normalize(torch.sign(text_out))
            regt = ((Bt * text_out).sum(1) - 1.).pow(2).sum()
            B1 = F.normalize(torch.sign(label_out))
            reg2 = ((B1 * label_out).sum(1) - 1.).pow(2).sum()
            regterm = reg1 + reg2 + regt

            loss_all = loss + loss_text + regterm * opt.beta + loss1 + loss_b * opt.alpha

            optimizer_hash.zero_grad()
            optimizer_label.zero_grad()
            optimizer_hash_text.zero_grad()
            loss_all.backward()
            optimizer_hash_text.step()
            optimizer_hash.step()
            optimizer_label.step()
            epoch_loss += loss_all.item() / the_batch
        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' %
              (epoch + 1, epochs, epoch_loss / len(train_loader)))
    cate_input = torch.eye(train_labels.size(1)).cuda()
    cate_code = label_model(cate_input).detach()
    anchor_code = torch.zeros((train_labels.size(0), code_length)).cuda()
    anchor_codet = torch.zeros((train_labels.size(0), code_length)).cuda()
    anchor_label = torch.zeros(
        (train_labels.size(0), train_labels.size(1))).cuda()
    for idx, (img, text, labelst, index) in enumerate(train_loader):
        img = img.cuda()
        labelst = labelst.type(torch.FloatTensor).cuda()
        text = text.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            code = hash_model(img, labelst)
            codet = hash_text_model(text, labelst)
            anchor_code[index] = code.detach()
            anchor_codet[index] = codet.detach()
            anchor_label[index] = labelst
    with open('./data_aware_proxy_codes/' + data_set + '_' + str(code_length) + '.pkl', 'wb') as f:
        pickle.dump({'image_code': anchor_code, 'text_code': anchor_codet, 'label': anchor_label, 'class_code': cate_code}, f)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.2, type=float, help='alpha')
    parser.add_argument('--beta', default=0.2, type=float, help='beta')
    parser.add_argument('--gamma', default=20, type=int, help='gamma')
    parser.add_argument('--sigma', default=0.2, type=float, help='sigma')
    parser.add_argument('--bit', default=32, type=int, help='hash code length')
    parser.add_argument('--dataset', default='iaprtc', type=str, help='dataset name')
    parser.add_argument('--data_path', default='iaprtc', type=str, help='dataset path')
    opt = parser.parse_args()
    DAPH_proxy_code_Algo(opt)

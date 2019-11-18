#! /usr/bin/env python

# Fully-connected network for classifying faces as people.

import numpy as np
import sys
from pympler import asizeof
import face_extraction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FCN(nn.Module):
    def __init__(self, num_class, in_size = 128):
        super(FCN, self).__init__()

        self.fc1 = nn.Linear(in_size, 500)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(500, 500)
        self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(500, 30)
        self.fc3 = nn.Linear(500, num_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = F.relu(self.fc2(x))
        x = self.relu2(x)
        x = self.fc3(x)
        return self.softmax(x)


def sort_common_faces(list_of_facerects, num_inst_thresh=150):
    assert isinstance(list_of_facerects, list)
    assert isinstance(list_of_facerects[0], face_extraction.FaceRect)

    ignored_names = ['.realignore', '.ignore']

    name_tagged_indices = [x for x in range(len(list_of_facerects)) if list_of_facerects[x].name not in ignored_names and list_of_facerects[x].name is not None]

    names = [list_of_facerects[x].name for x in name_tagged_indices]
    encodings = [list_of_facerects[x].encoding for x in name_tagged_indices]
    list_of_names = list(set(names)) 

    ignore_idcs = [x for x in range(len(list_of_facerects)) if list_of_facerects[x].name in ignored_names]
    ignore_encs = [list_of_facerects[x].encoding for x in ignore_idcs]

    # Test our break out with some assertions.
    name_idx = 0
    ign_idx = 0
    for x in range(min(len(list_of_facerects), 500)):
        if x in name_tagged_indices:
            assert list_of_facerects[x].name in list_of_names
            assert list_of_facerects[x].name not in ignored_names
            assert np.all(list_of_facerects[x].encoding == encodings[name_idx])
            name_idx += 1
        else:
            if x in ignore_idcs:
                assert list_of_facerects[x].name in ignored_names
                assert np.all(list_of_facerects[x].encoding == ignore_encs[ign_idx])
                ign_idx += 1

    # print(asizeof.asizeof(list_of_facerects[0].encoding))
    print(len(list_of_facerects))
    # ignored_encodings = [lis]
    # encodings = [x.encoding for x in list_of_facerects]

    # non_idx = list_of_names.index(None)
    # list_of_names.pop(non_idx)

    print(list_of_names)

    name_cnt = [names.count(x) for x in list_of_names]
    print(name_cnt)
    print(max(name_cnt))
    # Only use names that have more than a 
    # threshold of instances.
    names_thresh = [x >= num_inst_thresh for x in name_cnt]

    sufficient_names = []
    for i in range(len(names_thresh)):
        if names_thresh[i]:
            sufficient_names.append(list_of_names[i])

    suff_names_idcs = [x for x in range(len(list_of_facerects)) if list_of_facerects[x].name in sufficient_names]

    train_labels = [list_of_facerects[x].name for x in suff_names_idcs]
    train_encs = [list_of_facerects[x].encoding for x in suff_names_idcs]

    label_list = list(set(train_labels)) 
    print(label_list, len(train_encs))

    t = Variable(torch.tensor(train_encs[0]).unsqueeze(0))

    fcn = FCN(len(label_list)).double()

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    fcn.apply(init_weights)
    out = fcn(t)
    print(out, torch.sum(out))


    # Define a cross-entropy loss function
    # that works with soft labels. Source: 
    # https://stats.stackexchange.com/questions/206925/is-it-okay-to-use-cross-entropy-loss-function-with-soft-labels/215495
    def CXE(pred, lbl):
        # Numerical stability
        lbl += 0.000001
        pred += 0.000001
        # Normalize back to one, along the rows (dim),
        # with a one norm (p)
        lbl = F.normalize(lbl, dim=1, p=1)
        pred = F.normalize(pred, dim=1, p=1)

        log_x_pred = torch.log(pred)
        cost_val = -torch.sum(lbl * log_x_pred, dim=1)
        return cost_val
        # If p is distribution of the labels and
        # q is the distribution of the output of the
        # network, and we categorize values as a 
        # probability that a given class is true, then
        # we can cast the cross-entropy loss as : 
        # -p(y=0|x) --- (1-lbl)
        # * log(q(y=0 | x) ) --- log(1-pred)
        # - p(y=1|x) --- lbl
        # * log(q(y=1|x)) -- log(pred).
        # The more sure the labels are (i.e. one value
        # closer to 1), the lower the minimum loss will be.
        loss = ( (1-lbl) * torch.log(1-pred) - lbl * torch.log(pred) ).sum(dim=1)
        # assert loss > 0
        return loss

    # def CXE(predicted_sm, target_sm):
    #     return -(target_sm * torch.log(predicted_sm)).sum(dim=1).mean()

    label = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float()
    pred = torch.tensor([[0.979, 0.01, 0.01, 0.001]]).float()
    print(CXE(pred, label))
    label = torch.tensor([[0.57, .41, 0.01, 0.01]]).float()
    pred = torch.tensor([[.6, .38, 0.01, 0.01]]).float()
    # print(CXE(label, pred))
    print(CXE(pred, label))
    print(CXE(pred, pred))

    label = torch.tensor([[0.95, 0.025, 0.025, 0]]).float()
    pred = torch.tensor([[0.95, 0.025, 0.025, 0.001]]).float()
    print(".95:", CXE(pred, label))

    label = torch.tensor([[0.8, 0.1, 0.1, 0]]).float()
    pred = torch.tensor([[0.8, 0.1, 0.1, 0.0001]]).float()
    print(".8 ", CXE(pred, label))
    label = torch.tensor([[0.6, 0.2, 0.2, 0]]).float()
    pred = torch.tensor([[0.6, 0.2, 0.2, 0.001]]).float()
    print(".6 ", CXE(pred, label))
    label = torch.tensor([[0, 1.0, 0.0, 0.0]]).float()
    print(CXE(pred, label))
    label = torch.tensor([[.999, 0.0, 0.0, 0.0]]).float()
    print(CXE(pred, label))

    lbl = torch.tensor([[0, 1.0, 0.0, 0.0],
                          [0, 1.0, 0.0, 0.0]]).float()
    pred = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                         [0.1, 1.0, 0.0, 0.0]]).float()
    print(CXE(pred, lbl))
    lbl = torch.tensor([[ .25, .25, .25, .25 ],
                          [ .25, .25, .25, .25 ]]).float()
    pred = torch.tensor([[ .25, .25, .25, .25 ],
                         [ .0, 0, 0, .000001 ]]).float()
    print(CXE(pred, lbl))
    # print(CXE(label, label))

    # loss = nn.CrossEntropyLoss()
    # label = torch.tensor([[.7, .3, 0.0, 0.0]]).float()
    # label = torch.tensor([[1.0, .0, 0.0, 0.0]]).float()
    # print("Loss ", loss(label, torch.tensor([0])))
    # print("Loss ", loss(label, torch.tensor([1])))
# For a network trained with a label smoothing of parameter α, we minimize instead the cross-entropy between the modified targets y^LS_k and the networks’ outputs pk, where yLS k = yk(1 − α) + α/K
# When Does Label Smoothing Help paper
import argparse

import numpy as np
from numpy.linalg import inv, eigh

from data_loader import load_data
from utils import boolean_string, MultipleOptimizer, MultipleScheduler, find_gpu
from train_test import train_and_test

from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedKFold

import torch
from collections import Counter

class Network(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_layer, covlet=False):
        super(Network, self).__init__()

        layers = []
        if covlet == True:
            # layers += [torch.nn.ReLU()]
            layers += [torch.nn.ReLU(), torch.nn.Dropout(0.0)]
        if hidden_layer == True:
            layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU(), torch.nn.Dropout(0.)]
            layers += [torch.nn.Linear(in_size, out_size)]
        else:
            layers += [torch.nn.Linear(in_size, out_size)]
        self.laysers = torch.nn.Sequential(*layers)

    def forward(self, A0):
        x = self.laysers(A0)
        return x

class COVLET(torch.nn.Module):
    def __init__(self, w, v, num_feature, num_scale, power, device):
        super(COVLET, self).__init__()

        self.w = w
        self.v_T = v.T
        self.num_feature = num_feature
        self.num_scale = num_scale
        self.power = power
        self.device= device

        self.x_1 = 1.
        self.x_2 = 2.

        self.gamma_l = 1.3849001794597395
        
        self.eps = 1e-16

        self.K = torch.nn.Parameter(torch.FloatTensor(1,))
        self.K.data.uniform_(10.,100.)

        self.t = torch.nn.Parameter(torch.pow(10,torch.FloatTensor(self.num_scale,).uniform_(-0.8,1.3)))


    def g(self, x):
        alpha = 2.
        beta = 2.
        zero = torch.zeros(1,device=self.device)
        def s(x):
            return -5 + 11*x + -6*(x**2) + x**3

        kernel_value = torch.where(x < self.x_1, self.x_1**(-1*alpha) * x.pow(alpha), zero) \
                    + torch.where((x >= self.x_1) & (x <= self.x_2), s(x), zero) \
                    + torch.where(x > self.x_2, self.x_2**beta * (x+1e-12).pow(-1*beta), zero)

        return kernel_value

    def h(self, x):
        return torch.exp(-(x * self.K)**self.power)

    def forward(self, f_hat):
        CMD = []

        self.K.data.clamp_(min=self.eps)
        self.t.data.clamp_(min=self.eps)

        for i in range(self.num_scale):
            band_kernel_value = torch.diag(self.g(self.w * self.t[i]))
            CMD.append(f_hat @ band_kernel_value @ self.v_T)

        return torch.cat(CMD,dim=1)

class Items(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        assert(X.shape[0] == Y.shape[0])
        
        self.length  = X.shape[0]

        self.X = X
        self.Y = Y
        

    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        xx = torch.from_numpy(self.X[i])
        yy = torch.from_numpy(np.asarray(self.Y[i]))
        return xx, yy


def main(args):

    X, y, num_classes = load_data(args.DATA)
    device = torch.device(find_gpu() if torch.cuda.is_available() else "cpu")


    print(f"{args.DATA} DATA statistics: {Counter(y)}")

    num_samples = X.shape[0]
    num_feature = X.shape[1]

    mean_X = np.mean(X, axis = 0)     ## caculate mean of each features
    zm_X = X - mean_X    ## Make zero-mean for each features
    zm_X = zm_X.transpose()    ## Make matrix as same shape to Paper -> P x N
    cov_FA = np.matmul(zm_X,np.transpose(zm_X)) / num_samples     # Covariance Matrix
    # Use correlation instead
    diag = np.sqrt(np.diag(np.diag(cov_FA))); gaid = np.linalg.inv(diag); cor_FA = gaid @ cov_FA @ gaid
    w_FA, v_FA = eigh(cor_FA)     # Eigen values/vectors for covariance matrix

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    test_acc = []
    test_rec = []
    test_prc = []

    current_FOLD = 1

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(f"Training target statistics: {Counter(y_train)}")
        # print(f"Testing target statistics: {Counter(y_test)}")

        smote = ADASYN(random_state=RANDOM_STATE, sampling_strategy='minority')
        X_near, y_near = smote.fit_resample(X_train,y_train)

        X_hat_train = X_near @ v_FA
        X_hat_test = X_test @ v_FA

        max_epoch = 2000
        batch_size = 4096

        torch.autograd.set_detect_anomaly(True)
        
        if args.CMD == False:
            network_lr = 1e-3
            network_norm_rate = 1e-2

            train_loader = torch.utils.data.DataLoader(Items(X_near,y_near), batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(Items(X_test, y_test), batch_size=batch_size, shuffle=False)

            if args.MUL:
                model = Network(num_feature,num_classes,hidden_layer=True).to(device)
            else:
                model = Network(num_feature,num_classes,hidden_layer=False).to(device)
            
            covlet = None
            opt = torch.optim.AdamW(model.parameters(), lr=network_lr,weight_decay=network_norm_rate)
            network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(len(train_loader) * max_epoch))

        else: # When Covlet transform applied on
            num_scale = 32
            power = 2

            network_lr = 1e-2
            network_norm_rate = 1e-1
            scale_lr = 0
            scale_norm_rate = 0

            train_loader = torch.utils.data.DataLoader(Items(X_hat_train,y_near), batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(Items(X_hat_test, y_test), batch_size=batch_size, shuffle=False)
            
            if args.MUL:
                model = Network(num_feature*num_scale,num_classes,hidden_layer=True,covlet=True).to(device)
            else:
                model = Network(num_feature*num_scale,num_classes,hidden_layer=False,covlet=True).to(device)
        
            w = torch.from_numpy(w_FA).float().to(device)
            v = torch.from_numpy(v_FA).float().to(device)
            covlet = COVLET(w,v, num_feature, num_scale,power,device).to(device)

            network_optimizer = torch.optim.AdamW(model.parameters(), lr=network_lr, weight_decay=network_norm_rate)
            scale_optimizer = torch.optim.AdamW([list(covlet.parameters())[1]],lr=scale_lr, weight_decay=scale_norm_rate)
            opt = MultipleOptimizer(network_optimizer, scale_optimizer)
            
            network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(network_optimizer, T_max=(len(train_loader) * max_epoch))
        
        print(f"\n[{current_FOLD}-FOLD]\n")
        acc, rec, prc = train_and_test(max_epoch,train_loader,test_loader,model,opt,network_scheduler,device,covlet)
        test_acc.append(acc)
        test_rec.append(rec)
        test_prc.append(prc)
        print("\n")
        current_FOLD += 1
    
    print(test_acc)
    print("[5-FOLD AVERAGE] Acc: %.3f Rec: %.3f Prc: %.3f"%(np.mean(test_acc),np.mean(test_rec),np.mean(test_prc)))
    print("[5-FOLD Std] Acc: %.3f Rec: %.3f Prc: %.3f"%(np.std(test_acc),np.std(test_rec),np.std(test_prc)))


if __name__ == "__main__":
    RANDOM_STATE = 20220531
    torch.manual_seed(RANDOM_STATE)

    parser = argparse.ArgumentParser(description='Classification Type (2-way / 3-way) and Input Type (raw FA / CMD)')
    parser.add_argument('--DATA', default='fa', type=str, help='fa, eye, ct, cardio')
    parser.add_argument('--CMD', default=False, type=boolean_string, help='raw FA : False, CMD : True')
    parser.add_argument('--MUL', default=False, type=boolean_string, help='Multilayer layer NN : True, single layer NN : False')
    args = parser.parse_args()
    main(args)
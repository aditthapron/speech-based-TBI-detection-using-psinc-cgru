# This file is modified from https://github.com/mravanelli/SincNet

# To run this file: python Coelho.py --cfg Coelho.cfg --fold 0
# fold needs to be between 0-9

import glob
import os
import pandas as pd
import librosa
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from dnn_models import cGRU
from dnn_models import SincNet

from data_io import *
from sklearn.metrics import recall_score,precision_score,balanced_accuracy_score,f1_score,confusion_matrix,roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from skmultilearn.model_selection import IterativeStratification
import scipy

seed = 42
fold = 10
DEMOGRAPHICS_PATH = '/home/aditthapron/WASH_work/SincNet/all_demographic.csv'
DATA_PATH = '/home/aditthapron/WASH_work/TBIBank/tbi/English/Coelho/'
### Data loader for Coelho's corpus
class Dataset_(Dataset):
    def __init__(self, batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp,tran_lst):
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.wav_lst = wav_lst
        self.N_snt = N_snt
        self.wlen = wlen
        self.lab_dict = lab_dict
        self.fact_amp = fact_amp
        self.tran_lst  = tran_lst
        self.chunk_n = 20

    def __len__(self):
        return len(self.wav_lst)

    def __getitem__(self, idx):
        np.random.seed(int(idx))
        snt_id_arr=np.random.randint(self.N_snt)   
        rand_amp_arr = np.random.uniform(1.0-self.fact_amp,1+self.fact_amp,1)
        signal, fs = librosa.load(self.wav_lst[snt_id_arr],16000)
        signal=signal/np.max(np.abs(signal))
        select_ind = np.zeros(len(signal),dtype=bool)
        time = np.array(cha_read(self.tran_lst[snt_id_arr])).astype(int)
        time = (time*16000//1000)
        for t in time:
            select_ind[t[0]:t[1]]=1
        signal = signal[select_ind]
        signal = optimal_vtln(signal)

        snt_len=signal.shape[0] 
        snt_beg=np.random.randint(snt_len-self.wlen*self.chunk_n-1)
        snt_end=snt_beg+self.wlen*self.chunk_n
        channels = len(signal.shape)
        if channels == 2:
            print('WARNING: stereo to mono: ')
            signal = signal[:,0]
        
        sig_batch=signal[snt_beg:snt_end]*rand_amp_arr[0]
        if self.wav_lst[snt_id_arr].split('/')[-2] =='TB':
            lab_batch = 1 
        else:
            lab_batch = 0
        return sig_batch,lab_batch,self.wav_lst[snt_id_arr].split('/')[-1]


# Reading configuration file (modified from original SincNet)
options=read_conf()

#[data]
tr_lst=options.tr_lst
te_lst=options.te_lst
pt_file=options.pt_file
class_dict_file=options.lab_dict
data_folder=options.data_folder+'/'
output_folder=options.output_folder

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))

#[optimization]
lr = 0.003
batch_size = 4
N_epochs = 100
N_eval_epoch = 4



### set up 10 fold cross-validation
# the fold is determined by i
print("SEED:{} FOLD:{}".format(seed,options.fold))

tran_healthy = np.array(sorted(glob.glob(DATA_PATH + 'transcript/N/*.cha')))
healthy = np.array(sorted(glob.glob(DATA_PATH+'N/*.wav')))

tran_TBI = np.array(sorted(glob.glob(DATA_PATH + 'transcript/TB/*.cha')))
TBI = np.array(sorted(glob.glob(DATA_PATH+'TB/*.wav')))


#train/test split using IterativeStratification based on demographics

np.random.seed(seed)
healthy = np.delete(healthy,[23])
tran_healthy = np.delete(tran_healthy,[23])
TBI = np.delete(TBI,[26, 40, 51])
tran_TBI = np.delete(tran_TBI,[26, 40, 51])

#loading demographics files
df = pd.read_csv(DEMOGRAPHICS_PATH)
df = df.drop([26, 40, 51, 54+23])
X = np.array(df.iloc[:,0]).reshape(-1,1)
y=np.array(df.iloc[:,[1,2,3,5]]) # consider Age, Sex, Education, TBI(Y/N) for y label 
#Split age into two groups
med = np.median(y[:,0])
y[:,0]= [0 if i<med else 1 for i in y[:,0]]

#Encoding label and perform IterativeStratification
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
y=enc.transform(y)
val = []

# Train/Val/Test split
# Test set will not be used in the training.
# validation set is obtained in nested-cross validation fashion
k_fold = IterativeStratification(n_splits=fold, order=1)
j=0
tr_and_val=None
for train, test in k_fold.split(X, y): 
    ### iterative through k_fold.split(X, y) to get testing set (i th fold), randomly selecting one fold as validation and the rest for training
    if j==options.fold: #Determine if current fold is the testing set.
        tr_and_val=train
    j+=1
# For Training/Validation spliting

j=0
k_fold = IterativeStratification(n_splits=fold-1, order=1)
for train, test in k_fold.split(X[tr_and_val], y[tr_and_val]):
    if j==options.fold: #Determine if current fold is the testing set
        val=test
        tr=train
    j+=1

#Finalize files
wav_temp = np.hstack([TBI,healthy])
tran_temp = np.hstack([tran_TBI,tran_healthy])
wav_lst_tr = wav_temp[tr]
tran_lst_tr = tran_temp[tr]
wav_lst_te = wav_temp[val]
tran_lst_te = tran_temp[val]
snt_tr=len(wav_lst_tr)
snt_te=len(wav_lst_te)


# Folder creation
os.makedirs(output_folder, exist_ok=True) 
                
### ending of data preparation####
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.BCELoss()
cost_mfcc = nn.MSELoss()
m = nn.Sigmoid()
        
# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# Batch size for validation
Batch_dev=4


# Feature extractor Sinc
CNN_arch = {'input_dim': wlen,
    'fs': fs,
    'cnn_N_filt': cnn_N_filt,
    'cnn_len_filt': cnn_len_filt,
    'cnn_max_pool_len':cnn_max_pool_len,
    'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
    'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
    'cnn_use_laynorm':cnn_use_laynorm,
    'cnn_use_batchnorm':cnn_use_batchnorm,
    'cnn_act': cnn_act,
    'cnn_drop':cnn_drop,          
    }

CNN_net=SincNet(CNN_arch).to(device)
DNN1_net = cGRU(CNN_net.out_dim).to(device)

lab_dict=None

if pt_file!='none':
    checkpoint_load = torch.load(pt_file,device)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])


optimizer_CNN = None
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
scheduler_DNN1 = StepLR(optimizer_DNN1, step_size=6, gamma=0.99)


training_set = Dataset_(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.2,tran_lst_tr)
dataloader = DataLoader(training_set, batch_size=batch_size,shuffle=True, num_workers=5)
# dataloader = DataLoader(training_set, batch_size=2,shuffle=True, num_workers=2)
early_stop=0
best_loss=np.inf
fine_tuning=False
CHUNK=20

#GRU memory
Subject_h = dict()
for epoch in range(N_epochs):
    test_flag=0
    if optimizer_CNN == None:
        CNN_net.eval()
    else:
        CNN_net.train()
    DNN1_net.train()
    
    loss_sum=0
    err_sum=0
    c=0
    for local_batch, local_labels,subj  in dataloader:
        # Transfer to GPU
        inp, lab = local_batch.to(device).float(), local_labels.to(device)
        inp = inp.view(inp.shape[0]*CHUNK,wlen)
        feature = CNN_net(inp)
        feature = feature.view(inp.shape[0]//CHUNK,CHUNK,CNN_net.out_dim,CNN_net.out_filt_len).permute(0,2,1,3)
        feature = feature.reshape(inp.shape[0]//CHUNK,CNN_net.out_dim,CHUNK*CNN_net.out_filt_len).permute(0,2,1)
        
        #get memory
        h_in=torch.zeros((1, 4, 8)).to(device)
        for S in range(batch_size):
            if subj[S] in Subject_h:
                h_in[:,S,:] = Subject_h[subj[S]]
            else:
                #if new subject initilize to be zero
                h_in[:,S,:] = torch.zeros((1, 1, 8)).to(device)
        pout,h_out= DNN1_net(feature,h_in)
        #save memory
        for S in range(batch_size):
            Subject_h[subj[S]] = h_out[:,S,:]

        # loss calculation
        pred = torch.round(pout) 
        loss = cost(pout, lab.float())
        err = torch.mean((pred!=lab).float())

        #model optimization
        if fine_tuning:
            optimizer_CNN.zero_grad()
        optimizer_DNN1.zero_grad()      
        loss.backward(retain_graph=True)
        if fine_tuning:
            optimizer_CNN.step()
        optimizer_DNN1.step()

        loss_sum=loss_sum+loss.detach()
        err_sum=err_sum+err.detach()
        c=c+1

    loss_tot=loss_sum/c
    err_tot=err_sum/c
    if fine_tuning:
        scheduler_CNN.step()
    scheduler_DNN1.step()
    
 
    if epoch%N_eval_epoch==0:          
        CNN_net.eval()
        DNN1_net.eval()
        test_flag=1 
        loss_sum=0
        err_sum=0
        err_sum_snt=0
        
        with torch.no_grad():  
            acc=[]
            acc_2=[]
            prec = []
            recall = []
            pred_all = []
            label_all = []
            f1 = []
            auc = []
            sens = []
            loss_recon = 0
            Subject_h_val=dict()
            for i in range(snt_te):
                signal, fs = librosa.load(wav_lst_te[i],16000)
                subj = wav_lst_te[i].split('/')[-1]
                signal=signal/np.max(np.abs(signal))
                select_ind = np.zeros(len(signal),dtype=bool)
                time = np.array(cha_read(tran_lst_te[i])).astype(int)
                time = (time*16000//1000)
                for t in time:
                    select_ind[t[0]:t[1]]=1
                signal = signal[select_ind]
                
                # [signal, fs] = sf.read(data_folder+wav_lst_te[i])
                signal = optimal_vtln(signal)
                signal=torch.from_numpy(signal).float().to(device).contiguous()
                if tran_lst_te[i].split('/')[-2] =='TB':
                    lab_batch = 1 
                else:
                    lab_batch = 0
                # lab_batch=lab_dict[wav_lst_te[i]]
            
                # split signals into chunks
                beg_samp=0
                end_samp=wlen*CHUNK

                N_fr=int((signal.shape[0]-wlen*CHUNK)/(wshift))
                sig_arr=torch.zeros([Batch_dev,CHUNK*wlen]).float().to(device).contiguous()

                count_fr=0
                count_fr_tot=0
                temp=[]
                
                while end_samp<signal.shape[0]:
                    try:
                        sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                    except:
                        print(sig_arr.size())
                        print(wlen)
                        print(count_fr)
                    beg_samp=beg_samp+wshift*CHUNK
                    end_samp=beg_samp+wlen*CHUNK
                    count_fr=count_fr+1
                    count_fr_tot=count_fr_tot+1
                    if count_fr==Batch_dev:
                        inp=Variable(sig_arr)
                        temp_sig = sig_arr.detach().cpu().numpy()
                        mfcc =torch.from_numpy(np.array([librosa.feature.mfcc(b,16000,n_mels=40,n_mfcc=40) for b in temp_sig])).float().to(device).contiguous()
                        inp = inp.view(Batch_dev*CHUNK,wlen)
                        feature = CNN_net(inp)
                        feature = feature.view(Batch_dev,CHUNK,CNN_net.out_dim,CNN_net.out_filt_len).permute(0,2,1,3)
                        feature = feature.reshape(Batch_dev,CNN_net.out_dim,CHUNK*CNN_net.out_filt_len).permute(0,2,1)
                        
                        #get memory
                        h_in=torch.zeros((1, Batch_dev, 8)).to(device)
                        for S in range(Batch_dev):
                            if subj[S] in Subject_h_val:
                                h_in[:,S,:] = Subject_h_val[subj[S]]
                            else:
                                #if new subject initilize to be zero
                                h_in[:,S,:] = torch.zeros((1, 1, 8)).to(device)

                        pout,h_out = DNN1_net(feature)
                        #save memory
                        for S in range(Batch_dev):
                            Subject_h_val[subj[S]] = h_out[:,S,:]

                        pred = torch.round(pout)
                        lab= Variable((torch.zeros(Batch_dev)+lab_batch).to(device).contiguous().float())
                        loss = cost(pout, lab.float())
                        err = torch.mean((pred!=lab.float()).float())
                        loss_sum = loss_sum+loss.detach()/N_fr
                        err_sum=err_sum+err.detach()/N_fr
                        
                        pred_all.append(pred.detach().cpu().numpy())
                        label_all.append(lab.detach().cpu().numpy())
                        count_fr=0
                        sig_arr=torch.zeros([Batch_dev,CHUNK*wlen]).float().to(device).contiguous()
            
            loss_recon = loss_recon/i
            acc.append(balanced_accuracy_score(np.hstack(label_all),np.hstack(pred_all)))
            recall.append(recall_score(np.hstack(label_all),np.hstack(pred_all)))
            prec.append(precision_score(np.hstack(label_all),np.hstack(pred_all)))
            f1.append(f1_score(np.hstack(label_all),np.hstack(pred_all)))
            auc.append(roc_auc_score(np.hstack(label_all),np.hstack(pred_all)))
            err_tot_dev_snt=0
            loss_tot_dev=loss_sum/snt_te
            err_tot_dev=err_sum/snt_te

        print("epoch %i, loss_tr=%f loss_te=%f recon=%f" % (epoch, loss_tot,loss_tot_dev,loss_recon))
        print("ACC:{}\tACC:{}\tPrec:{}\tRecall:{}\tF1:{}\tAUC:{}".format(np.mean(acc),np.mean(acc_2),np.mean(prec),np.mean(recall),np.mean(f1),np.mean(auc)))
        with open(output_folder+"/res_{}_{}.res".format(str(options.fold),str(fold)), "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))   
            res_file.write("ACC:{}\tPrec:{}\tRecall:{}\tF1:{}\tAUC:{}".format(np.mean(acc),np.mean(prec),np.mean(recall),np.mean(f1),np.mean(auc)))
        if loss_tot_dev < best_loss:
            checkpoint={'CNN_model_par': CNN_net.state_dict(),
                        'DNN1_model_par': DNN1_net.state_dict()
                        }
            torch.save(checkpoint,output_folder+'/model_raw_{}_{}.pkl'.format(str(options.fold),str(fold)))
            early_stop=0
            best_loss = loss_tot_dev
        else:
            early_stop += 1

        #begin fine-tuning
        if ~fine_tuning and early_stop==10:
            fine_tuning=True
            early_stop=0
            fine_tuning_epoch=epoch

            for param in CNN_net.parameters():
                param.requires_grad = True

            #set early stage of fine tuning
            optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=1e-5,alpha=0.95, eps=1e-8) 
            optimizer_DNN = optim.RMSprop(DNN_net1.parameters(), lr=1e-5,alpha=0.95, eps=1e-8) 
            scheduler_CNN = StepLR(optimizer_CNN, step_size=1, gamma=6.31) 
            scheduler_DNN = StepLR(optimizer_DNN, step_size=1, gamma=6.31)

        if epoch == fine_tuning_epoch+10:
            #return to normal learning rate
            optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=1e-3,alpha=0.95, eps=1e-8) 
            optimizer_DNN = optim.RMSprop(DNN_net1.parameters(), lr=1e-3,alpha=0.95, eps=1e-8) 
            scheduler_CNN = StepLR(optimizer_CNN, step_size=1, gamma=0.95) 
            scheduler_DNN = StepLR(optimizer_DNN1, step_size=1, gamma=0.95)
        #Early-stopping execution
        if fine_tuning and early_stop==20:
            break
    
    
    else:
        print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot,err_tot))



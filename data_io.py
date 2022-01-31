import configparser as ConfigParser
from optparse import OptionParser
import numpy as np
#import scipy.io.wavfile
import torch

def cha_read(file):
    # text = []
    time = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('*'):
                try:
                    if line.find('PAR')!=-1 and line.find('\x15')!=-1:
                        # text.append(line.split('\x15')[0].split('PAR:\t')[1])
                        time.append(line.split('\x15')[1])
                except:
                    pass
    return [s.split('_') for s in time]


def optimal_vtln(Y, warpFunction='symmetric'):
    min_mse = np.inf
    try:
        X = np.load('baseline.npy')
    except:
        print('Please create a baseline audio file for VTLN in baseline.npy. Skipping VTLN')
        return Y
    for alpha in np.arange(0.1,1.8,.1):
        Xhat = vtln(Y, warpFunction, alpha)
        clip = min(Xhat.shape[1],X.shape[1])
        mse = ((X[:,:clip] - Xhat[:,:clip])**2).mean()
        if mse < min_mse:
            min_mse = mse
            min_alpha = alpha
    print(min_alpha)
    return vtln(Y, warpFunction, min_alpha)

def vtln(frames, warpFunction='asymmetric', alpha=1.2):
    frames = frames.T
    warp_funs = ['asymmetric', 'symmetric', 'power', 'quadratic', 'bilinear']
    if not warpFunction in warp_funs:
        print("Invalid warp function")
        return
    warpedFreqs = np.zeros(frames.shape)
    for j in range(len(frames)):
        m = len(frames[j])
        omega = (np.arange(m)+1.0) / m * np.pi
        omega_warped = omega
        if warpFunction is 'asymmetric' or warpFunction is 'symmetric':
            omega0 = 7.0/8.0 * np.pi
            if warpFunction is 'symmetric' and alpha > 1:
                omega0 = 7.0/(8.0 * alpha) * np.pi
            omega_warped[np.where(omega <= omega0)] = alpha * omega[np.where(omega <= omega0)]
            omega_warped[np.where(omega > omega0)] = alpha * omega0 + ((np.pi - alpha * omega0)/(np.pi - omega0)) * (omega[np.where(omega > omega0)] - omega0)
            omega_warped[np.where(omega_warped >= np.pi)] = np.pi - 0.00001 + 0.00001 * (omega_warped[np.where(omega_warped >= np.pi)])
        elif warpFunction is 'power':
            omega_warped = np.pi * (omega / np.pi) ** alpha
        elif warpFunction is 'quadratic':
            omega_warped = omega + alpha * (omega / np.pi - (omega / np.pi)**2)
        elif warpFunction is 'bilinear':
            z = np.exp(omega * 1j)
            omega_warped = np.abs(-1j * np.log((z - alpha)/(1 - alpha*z)))
        omega_warped = omega_warped / np.pi * m
        warpedFrame = np.interp(omega_warped, np.arange(m)+1, frames[j]).T
        if np.isreal(frames[j][-1]):
            warpedFrame[-1] = np.real(warpedFrame[-1])
        warpedFrame[np.isnan(warpedFrame)] = 0
        warpedFreqs[j]=warpedFrame
    return warpedFreqs.T

def stft_pad(frames):
    n = len(frames)
    n_fft = 512
    y_pad = librosa.util.fix_length(frames, n + n_fft // 2)
    return librosa.stft(y_pad, n_fft=n_fft),n


def ReadList(list_file):
    f=open(list_file,"r")
    lines=f.readlines()
    list_sig=[]
    for x in lines:
        list_sig.append(x.rstrip())
    f.close()
    return list_sig


def read_conf():
    parser=OptionParser()
    parser.add_option("--cfg", type="string") # Mandatory
    parser.add_option("--fold", type="int")
    (options,args)=parser.parse_args()
    cfg_file=options.cfg
    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)

    #[data]
    options.tr_lst=Config.get('data', 'tr_lst')
    options.te_lst=Config.get('data', 'te_lst')
    options.lab_dict=Config.get('data', 'lab_dict')
    options.data_folder=Config.get('data', 'data_folder')
    options.output_folder=Config.get('data', 'output_folder')
    options.pt_file=Config.get('data', 'pt_file')

    #[windowing]
    options.fs=Config.get('windowing', 'fs')
    options.cw_len=Config.get('windowing', 'cw_len')
    options.cw_shift=Config.get('windowing', 'cw_shift')

    #[cnn]
    options.cnn_N_filt=Config.get('cnn', 'cnn_N_filt')
    options.cnn_len_filt=Config.get('cnn', 'cnn_len_filt')
    options.cnn_max_pool_len=Config.get('cnn', 'cnn_max_pool_len')
    options.cnn_use_laynorm_inp=Config.get('cnn', 'cnn_use_laynorm_inp')
    options.cnn_use_batchnorm_inp=Config.get('cnn', 'cnn_use_batchnorm_inp')
    options.cnn_use_laynorm=Config.get('cnn', 'cnn_use_laynorm')
    options.cnn_use_batchnorm=Config.get('cnn', 'cnn_use_batchnorm')
    options.cnn_act=Config.get('cnn', 'cnn_act')
    options.cnn_drop=Config.get('cnn', 'cnn_drop')


    return options


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError 
         
         

import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/S2-KERNEL-METHODS/code_report")

import numpy as np
from itertools import product #, combinations
import pandas as pd
#import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import Levenshtein
import time
from tqdm import tqdm


### Lodding data #####
def load_sequence(kind='X', root='tr', number=3):
    """
    Load DNA sequences
    """
    seqs =  [pd.read_csv('./data/%s%s%d.csv'%(kind, root, d)) for d in range(number)]
    
    if kind == 'X':
            df = pd.DataFrame(columns=['Id','seq'])
    else:
            df= pd.DataFrame(columns=['Id','Bound'])
    
    for seq in seqs:
        
        df = df.append(seq, ignore_index=True)
        
    return df

def load_features(root='tr', number=3):
    """
    Load precomputed features
    """
    kind = 'X'
    
    feats =  [np.loadtxt('./data/%s%s%d_mat100.csv'%(kind, root, d)) for d in range(number)]
        
    return  np.vstack((feat for feat in feats))


####### k-spectrum  and variants #############
def getKmers(sequence, size=5):
    """
    Builds kmers
    """
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


def get_features(dF, size=5, normed=False, rang=(4,4)):
    """
    Compute DNA n_grams embeddings
    """
    df = dF.copy()
    
    df['words'] = df.apply(lambda x: getKmers(x['seq'], size=size), axis=1)
    df = df.drop('seq', axis=1)

    texts = list(df['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(df.iloc[item,1])
    
    if normed:
        cv = TfidfVectorizer(ngram_range=rang)
    else:
        cv = CountVectorizer(ngram_range= rang)
    X = cv.fit_transform(texts)
    return X

def save_spectrum_kernels(df, dg,  k=0, sizes=[3,4,5,6,7], rang=(4,4)): 
    tt = time.time()
    
    for size in sizes:
        print('Doing size: ', size)
        dF = df.iloc[2000*k:2000*(k+1)].append(dg.iloc[1000*k:1000*(k+1)])
        X = get_features(dF, size=size, normed=False, rang=rang)
        K = np.dot(X, X.T).toarray()
        np.savetxt('./mykernels/spectrum/K_%d_%d.txt'%(k, size), K)
        
    tt = time.time() - tt
    print('done is %.3f seconds'%(tt/60))

def get_exp_mismatch_matrix(words, _lambda):
    N = len(words)

    exp_mismatch_matrix = np.zeros((N, N))
    for i in range(N):
        exp_mismatch_matrix[i,i] = 1
        for j in range(i+1, N):
            exp_mismatch_matrix[i,j] = _lambda**Levenshtein.hamming(words[i], words[j])
            exp_mismatch_matrix[j,i] = exp_mismatch_matrix[i,j]

    return exp_mismatch_matrix

def get_exp_mismatch(dF, size=5, normed=False, gamma=0.4):
    df = dF.copy()
    
    df['words'] = df.apply(lambda x: getKmers(x['seq'], size=size), axis=1)
    df = df.drop('seq', axis=1)

    texts = list(df['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(df.iloc[item,1])
    
    if normed:
        cv = TfidfVectorizer()
    else:
        cv = CountVectorizer()
    X = cv.fit_transform(texts)
    
    words = list(cv.get_feature_names())
    S = get_exp_mismatch_matrix(words, gamma)
    
    K = X @ S @ X.T
    
    return K


def save_exponential_kernels(df, dg,  k=0, sizes=[3,4,5,6,7], gamma=0.4):
    tt = time.time()
    
    for size in sizes:
        print('Doing size: ', size)
        dF = df.iloc[2000*k:2000*(k+1)].append(dg.iloc[1000*k:1000*(k+1)])
        K = get_exp_mismatch(dF, size=size, normed=False, gamma=gamma)
        np.savetxt('./mykernels/exponential/K_%d_%d_%d.txt'%(k, size, 10*gamma), K)


    tt = time.time() - tt
    print('done is %.3f seconds'%(tt/60))

######
def get_phi_km(x, k, m, betas):
    """
    Compute feature vector of sequence x for Mismatch Kernel (k,m)
    :param x: string, DNA sequence
    :param k: int, length of k-mers
    :param m: int, maximal mismatch
    :param betas: list, all combinations of k-mers drawn from 'A', 'C', 'G', 'T'
    :return: np.array, feature vector of x
    """
    phi_km = np.zeros(len(betas))
    for i in range(101 - k + 1):
        kmer = x[i:i + k]
        for i, b in enumerate(betas):
            phi_km[i] += (np.sum(kmer != b) <= m)
    return phi_km


def letter_to_num(x):
    """
    Replace letters by numbers
    :param x: string, DNA sequence
    :return: string, DNA sequence with numbers instead of letters
    """
    x_str =  x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4')
    return np.array(list(x_str)).astype(int)


def get_true_mismatch(df, dg, ind=0, k=5, m=1, build=True):

    
    betas = np.array([letter_to_num(''.join(c)) for c in product('ACGT', repeat=k)])

    dF = df.iloc[2000*ind:2000*(ind+1)].append(dg.iloc[1000*ind:1000*(ind+1)])
    nf = dF.shape[0]
    
    if build:
        phi_km = np.zeros((nf, len(betas)))
    
        for i in tqdm(range(nf)):
            x = letter_to_num(dF.seq[i])
            phi_km[i] = get_phi_km(x, k, m, betas)
        np.save('./mykernels/mismatch/phi_km_%d_%d_%d.npy'%(ind, k, m), phi_km)
        
    else:
        phi_km= np.load('./mykernels/mismatch/phi_km_%d_%d_%d.npy'%(ind, k, m))
        
    K = np.dot(phi_km, phi_km.T)
    
    return K

#==============================================================================#



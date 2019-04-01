from kernels import load_sequence, save_spectrum_kernels, save_exponential_kernels, get_true_mismatch
import numpy as np
import pandas as pd
from classifiers import MultiKerOpt, CvSearch, get_best

sequences_train = load_sequence(number=3)
sequences_test = load_sequence(number=3, root='te')
labels_train = load_sequence(kind='Y' ,root='tr', number=3)


all_labels = labels_train.Bound.values.astype(int)

def _normalize(K, normalize=True):
    if normalize:
        diag = np.sqrt(np.diag(K))[:, np.newaxis]
        K_ = (1/diag) * K * (1/diag.T)
        k_xx, k_yx = K_[:2000, :2000], K_[2000:,:2000]
    else:
        k_xx, k_yx = K[:2000, :2000], K[2000:,:2000] 
    return k_xx, k_yx

def build_training(k, sizes=[4,5], gamma=0.4, spectrum=False, m=1, normalize=True):

    K_xx, K_yx = [], []
    
    for size in sizes:
        if spectrum:
            print('loading spectrum kernel for size %d '%size)
            K = np.loadtxt('./mykernels/spectrum/K_%d_%d.txt'%(k, size))
            k_xx, k_yx = _normalize(K, normalize=normalize)

            K_xx.append(k_xx)
            K_yx.append(k_yx)
        
        if gamma is not None:
            print('loading exponential kernel for size %d '%size)
            K = np.loadtxt('./mykernels/exponential/K_%d_%d_%d.txt'%(k, size, 10*gamma))
            k_xx, k_yx = _normalize(K, normalize=normalize)
            
            K_xx.append(k_xx)
            K_yx.append(k_yx)
            
        if m is not None and size in [4,5]:
            print('loading mismatch kernels for size %d '%size)
            K = get_true_mismatch(sequences_train, sequences_test, ind=k, k=size, m=1, build=False)
            k_xx, k_yx = _normalize(K, normalize=normalize)
            
            K_xx.append(k_xx)
            K_yx.append(k_yx)
            
    
    y_train = all_labels[2000*k:2000*(k+1)]
    
    return np.array(K_xx), np.array(K_yx), y_train

#==========Building Kernels =================#
sizes =  [3, 4, 5, 6, 7]
YET_TO_SAVE_SPECTRUM = True
if YET_TO_SAVE_SPECTRUM:
    save_spectrum_kernels(sequences_train, sequences_test, k=0, sizes=sizes, rang=(4,4))
    save_spectrum_kernels(sequences_train, sequences_test, k=1, sizes=sizes, rang=(4,4))
    save_spectrum_kernels(sequences_train, sequences_test, k=2, sizes=sizes, rang=(4,4))
    
YET_TO_SAVE_EXPONENTIAL = True

if YET_TO_SAVE_EXPONENTIAL:
    
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
        print(gamma)
        save_exponential_kernels(sequences_train, sequences_test, k=0, sizes=sizes, gamma=gamma)
        save_exponential_kernels(sequences_train, sequences_test, k=1, sizes=sizes, gamma=gamma)
        save_exponential_kernels(sequences_train, sequences_test, k=2, sizes=sizes, gamma=gamma)



YET_TO_SAVE_MISMATCH = True
if YET_TO_SAVE_MISMATCH:
    Kk = get_true_mismatch(sequences_train, sequences_test, ind=0, k=3, m=1, build=False)
    Kk = get_true_mismatch(sequences_train, sequences_test, ind=1, k=3, m=1, build=False)
    Kk = get_true_mismatch(sequences_train, sequences_test, ind=2, k=3, m=1, build=False)


    Kk = get_true_mismatch(sequences_train, sequences_test, ind=0, k=4, m=1, build=False)
    Kk = get_true_mismatch(sequences_train, sequences_test, ind=1, k=4, m=1, build=False)
    Kk = get_true_mismatch(sequences_train, sequences_test, ind=2, k=4, m=1, build=False)


    Kk = get_true_mismatch(sequences_train, sequences_test, ind=0, k=5, m=1, build=False)
    Kk = get_true_mismatch(sequences_train, sequences_test, ind=1, k=5, m=1, build=False)
    Kk = get_true_mismatch(sequences_train, sequences_test, ind=2, k=5, m=1, build=False)
#====================================================================###

REG_PARAMS_SPAN = [10**i for i in range(-3, 2)]
GAMMA = 0.1

LAUNCH = True

if LAUNCH:
    print(GAMMA)

    K_xx_0, K_yx_0, y_train_0 = build_training(0, sizes=[3,4,5,6,7],  gamma=GAMMA, spectrum=True, m=1, normalize=True)
    df  = CvSearch(K_xx_0, K_yx_0, y_train_0, method='svm', degrees=[1, 2, 3], 
               alphas=REG_PARAMS_SPAN, cv=5)
    df.to_csv('./summary/global/X0_normed_%d.csv'%(10*GAMMA), index=False)
    best_degree_0, best_alpha_0, best_0 = get_best(df)
    
    
    K_xx_1, K_yx_1, y_train_1 = build_training(1, sizes=[3,4,5,6,7],  gamma=GAMMA, spectrum=True, m=1, normalize=True)
    df  = CvSearch(K_xx_1, K_yx_1, y_train_1, method='svm', degrees=[1, 2, 3], 
               alphas=REG_PARAMS_SPAN, cv=5)
    df.to_csv('./summary/global/X1_normed_%d.csv'%(10*GAMMA), index=False)
    best_degree_1, best_alpha_1, best_1 = get_best(df)
    
    K_xx_2, K_yx_2, y_train_2 = build_training(2, sizes=[3,4,5,6,7],  gamma=GAMMA, spectrum=True, m=1, normalize=True)
    df  = CvSearch(K_xx_2, K_yx_2, y_train_2, method='svm', degrees=[1, 2, 3], 
               alphas=REG_PARAMS_SPAN, cv=5)
    df.to_csv('./summary/global/X2_normed_%d.csv'%(10*GAMMA), index=False)
    best_degree_2, best_alpha_2, best_2 = get_best(df)
    
    clf0 = MultiKerOpt(alpha=best_alpha_0, tol=1e-07, degree=best_degree_0, method='svm', hide=False)
    clf0.fit(K_xx_0, y_train_0, n_iter=5)
    y_pred_0 = clf0.predict(K_yx_0)
    
    clf1 = MultiKerOpt(alpha=best_alpha_1, tol=1e-07, degree=best_degree_1, method='svm')
    clf1.fit(K_xx_1, y_train_1, n_iter=5)
    y_pred_1 = clf1.predict(K_yx_1)
    
    clf2 = MultiKerOpt(alpha=best_alpha_2, tol=1e-07, degree=best_degree_2, method='svm')
    clf2.fit(K_xx_2, y_train_2, n_iter=5)
    y_pred_2 = clf2.predict(K_yx_2)
    
    y_pred = np.hstack((y_pred_0, y_pred_1, y_pred_2)).astype(int)
    Ids = np.array(range(3000))
    predictions = pd.DataFrame({'Id':Ids, 'Bound':y_pred.flatten()})
    
    predictions.to_csv('./predictions/Yte.csv', index=False)







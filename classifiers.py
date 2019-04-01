#import cvxopt
from cvxopt import matrix, spmatrix, solvers
#import scipy
from scipy.special import expit
import numpy as np
import sys
import pandas as pd
import time

class KernelNC():
    """
    distance based classifier for spectrum kernels
    """
    
    def __init__(self, classes):
        self.classes = classes
    
    def compute_dist(self, X, Y):
        K_x = np.dot(X, X.T).toarray()
        K_y = np.dot(Y, Y.T).toarray()
        K_xy = np.dot(X, Y.T).toarray()
        
        return np.diag(K_x) - 2*K_xy.mean(axis=1) + K_y.mean()
    
    def predict(self, X):
        
        dists = np.array([self.compute_dist(X, classe) for classe in self.classes])
        return dists.argmin(axis=0)
    
    def score(self, X, y):
        y__ = self.predict(X)
        return 100*(y__==y).mean()

class MultiKerOpt():
    
    def __init__(self, alpha=0.01, tol=1e-07, degree=2, method='klr', hide=False):
        self.alpha = alpha
        self.tol = tol
        self.degree = degree
        self.method = method
        self.hide  = hide
        
    def scale(self, u, norm):
        if norm=='l1':
            return u/np.sum(u)
        elif norm=='l2':
            return u / np.sqrt(np.sum(u**2))
        else:
            raise Exception('l1 and l2 are the only available norms')
            
    def bound(self, u, u_0, gamma, norm):
        u__ = u - u_0
        u__ = np.abs(self.scale(u__, norm) * gamma)
        return u__ + u_0
    
    def KrrIterate(self, Kernels, y, coef, weights = None):
        """
        Weighted KRR iterations
        """
        K_w = np.sum((Kernels * coef[:, None, None]), axis=0) ** self.degree
        N, D = K_w.shape
        if weights is None:
            c = np.linalg.solve(np.linalg.inv(K_w + self.alpha * np.eye(N, D)), y[:, np.newaxis])
        else:
            W_r = np.diag(np.sqrt(weights))
            A = W_r.dot(K_w).dot(W_r) + self.alpha * np.eye(N,D)
            Y = np.dot(W_r, y[:, np.newaxis])
            x_sol = np.linalg.solve(A, Y)
            c = np.dot(W_r, x_sol)
        return c
    
    def KlrIterate(self, Kernels, y, coef, tol=1e-07, max_iters=5):
        """
        KLR iterations
        """
        c_old = self.KrrIterate(Kernels, y, coef)
        K_w = np.sum((Kernels * coef[:, None, None]), axis=0) ** self.degree
        y_enc = 2*y-1
        
        for i in range(max_iters):
            m_t = np.dot(K_w, c_old)
            p_t = -expit(-y_enc[:, np.newaxis]*m_t)
            w_t = expit(m_t)*expit(-m_t)
            z_t = m_t - (p_t * y_enc[:, np.newaxis]) /(w_t+ 1e-05)
            c_new = self.KrrIterate(Kernels, z_t.flatten(), coef, weights=w_t.flatten())
            if np.linalg.norm(c_new - c_old)<tol:
                break
            else:
                c_old = c_new
        return c_old

    def SvmIterate(self, Kernels, y, coef):
        """
        SVM Estimation
        """
        nb_samples = y.shape[0]
        C = 1 / ( 2 * self.alpha * nb_samples)
        
        r = np.arange(nb_samples)
        o = np.ones(nb_samples)
        z = np.zeros(nb_samples)
            
        K_w  = np.sum(Kernels * coef[:, None, None], axis=0) ** (self.degree)
        
        y_enc = 2*y-1
        
        P = matrix(K_w.astype(float), tc='d')
        q = matrix(-y_enc, tc='d')
        G = spmatrix(np.r_[y_enc, -y_enc], np.r_[r, r + nb_samples], np.r_[r, r], tc='d')
        h = matrix(np.r_[o * C, z], tc='d')
        
        if self.hide:
            solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)
        c = np.ravel(sol['x'])[:,np.newaxis]
        
        return c
    
    def gradUpdate(self, Kernels, coef, delta):
        """
        Updating Gradient
        """
        K_t = np.sum(Kernels * coef[:, None, None], axis=0) ** (self.degree-1)
        grad = np.zeros(len(Kernels))
        for m in range(len(Kernels)):
            grad[m] = delta.T.dot((K_t * Kernels[m])).dot(delta)
            
        return - self.degree * grad
    
    def fit(self, Kernels, y, u_0=0, gamma=1, norm='l2', n_iter=5, step=1, weights=None):
        coef = np.random.normal(0, 1, len(Kernels)) / len(Kernels)
        coef = self.bound(coef, u_0, gamma, norm)
        new_coef = 0
        
        score_prev = np.inf
        
        for i in range(n_iter):
            #print(i+1)
            if self.method=='klr':
                delta = self.KlrIterate(Kernels, y, coef, tol=1e-07, max_iters=5)
            elif self.method=='svm':
                delta = self.SvmIterate(Kernels, y, coef)
            else:
                delta = self.KrrIterate(Kernels, y, coef, weights = weights)
                
            grad = self.gradUpdate(Kernels, coef, delta)
            
            new_coef = coef - step * grad
            new_coef = self.bound(new_coef, u_0, gamma, norm)
            
            score = np.linalg.norm(new_coef - coef, np.inf)
            
            if score>score_prev:
                step *= 0.9
                
            if score<self.tol:
                self.coef = coef
                self.delta = delta
            
            coef = new_coef
            score_prev = score.copy()
            
        self.coef, self.delta = coef, delta
        #return new_coef
    def predict(self, Kernels):
        K_w = np.sum(Kernels * self.coef[:, None, None], axis=0) ** (self.degree)
        y__ = np.sign(K_w.dot(self.delta)).flatten()
        if self.method != 'krr':
            y__ = 0.5 * (y__ + 1)
        return y__
    
    def score(self, Kernels, y):
        y__ = self.predict(Kernels)
        if self.method!='krr':
            score = 100*(y__==y).mean()
        else:
            score = np.mean((y__- y)**2)
        return score
                
    
def CvSearch(K_xx, K_yx, y, method='svm', degrees=[4], alphas=[0.01], cv=5, n_iter=5):
    tt = time.time()
    
    n_iters = cv * len(degrees) * len(alphas)
    
    n_samples = y.shape[0]
    
    DEG, ALPH, TRAIN, VAL = [], [], [], []
    
    i=0
    
    for degree in degrees:
        for alpha in alphas:
            DEG.append(degree)
            ALPH.append(alpha)
            
            #SPLITTING
            INDS = np.array(range(n_samples))
            idx = np.random.permutation(n_samples)
            INDS = INDS[idx]
            
            vals = np.array_split(INDS, cv)
            
            perfs_train = []
            perfs_val = []
            
            for val in vals:
                i += 1 
                sys.stderr.write('\rIteration %d/%d -- degree %d --alpha %.3f' %(i, n_iters, degree, alpha))
                sys.stderr.flush()
                
                train = np.setdiff1d(range(n_samples),val)
                
                clf = MultiKerOpt(alpha=alpha, tol=1e-07, degree=degree, method=method, hide=True)
                
                clf.fit(K_xx[:,train.reshape(-1,1), train], y[train], n_iter=n_iter)
                
                score_train = clf.score(K_xx[:,train.reshape(-1,1), train], y[train])
                
                score_val =  clf.score(K_xx[:,val.reshape(-1,1), train], y[val])
                
                perfs_train.append(score_train)
                perfs_val.append(score_val)
                
            TRAIN.append(np.mean(np.array(perfs_train)))
            VAL.append(np.mean(np.array(perfs_val)))
            
    df = pd.DataFrame({'degree':DEG, 'alpha':ALPH, 'train':TRAIN, 'val':VAL})
    
    tt = time.time() - tt
    print('Done in %.3f'%(tt/60))
    
    return df
#
def get_best(df):
    idx = np.argmax(df.val.values)
    best = np.max(df.val.values)

    best_degree = df.degree[idx]
    best_alpha = df.alpha[idx]
    return best_degree, best_alpha, best

import numpy as np

Y = ['F','L','I','M','V','S','P','T','A','Y','ST','H','Q','N','K','D','E','C','W','R','G']
X = ['UUU', 'UUC', 'UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG', 'AUU', 'AUC', 'AUA', 'AUG', 'GUU', 'GUC', 'GUA', 'GUG', 'UCU', 'UCC', 'UCA', 'UCG', 'CCU', 'CCC', 'CCA', 'CCG', 'ACU', 'ACC', 'ACA', 'ACG', 'GCU', 'GCC', 'GCA', 'GCG', 'UAU', 'UAC', 'UAA', 'UAG', 'CAU', 'CAC', 'CAA', 'CAG', 'AAU', 'AAC', 'AAA', 'AAG', 'GAU', 'GAC', 'GAA', 'GAG', 'UGU', 'UGC', 'UGA', 'UGG', 'CGU', 'CGC', 'CGA', 'CGG', 'AGU', 'AGC', 'AGA', 'AGG', 'GGU', 'GGC', 'GGA', 'GGG']


ITERATION = 10           # maximum # of iterations

R = np.zeros((21,ITERATION))  # pmf of Y
T = np.zeros((64,ITERATION))  # helper function
Q = np.zeros((64,ITERATION))  # pmf of X

mat = 0.9999
na = 0.000005

W = np.zeros((64,21))

W[:,:] = na
W[0,0] = mat
W[1,0] = mat
W[2:8,1] = mat
W[2:8,1] = mat
W[8:11,2] = mat
W[11,3] = mat
W[12:16,4] = mat
W[16:20,5] = mat
W[56:58,5] = mat
W[20:24,6] = mat
W[24:28,7] = mat
W[28:32,8] = mat
W[32:34,9] = mat
W[34:36,10] = mat
W[50,10] = mat
W[36:38,11] = mat
W[38:40,12] = mat
W[40:42,13] = mat
W[42:44,14] = mat
W[44:46,15] = mat
W[46:48,16] = mat
W[48:50,17] = mat
W[51,18] = mat
W[52:56,19] = mat
W[58:60,19] = mat
W[60:64,20] = mat

#print(W)

Q_init = np.zeros(64)
Q_init[:] = 1/64

#print(Q_init)

def Rr(x, W):
    return np.dot(x,W)

def Tr(x, W, r):
    X = np.repeat(x.reshape(1,-1), 64, axis=0)
    return np.sum(W*np.log(X*W/r), axis=0)

def Qr1(t):
    return np.exp(t) / np.sum(np.exp(t))

def converge(t, q, eps):
    return np.amax(t-np.log(q)) - np.amin(t-np.log(q)) < eps


Q[:,0] = Q_init
for i in range(ITERATION+1):
    R[:,i] = Rr(Q[:,i], W)
    print(np.sum(R))
    T[:,i] = Tr(Q[:,i], W, R[:,i])
    #print(T)
    Q[:,i+1] = Qr1(T[:,i])
    #print(Q)
    if converge(T[:,i], Q[:,i], 0.001):
        print("Convergence at ", i, " iterations!")
        break

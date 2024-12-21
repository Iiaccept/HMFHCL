import numpy as np

def TCMF1(alpha, Y, maxiter,A,B,C):
    iter0=1
    while True:

        a = np.dot(Y,B)
        b = np.dot(np.transpose(B),B)+alpha*C
        A = np.dot(a, np.linalg.inv(b))
        c = np.dot(np.transpose(Y),A)
        d = np.dot(np.transpose(A), A) + alpha * C
        B = np.dot(c, np.linalg.inv(d))

        if iter0 >= maxiter:
            #print('reach maximum iteration!')
            break
        iter0 = iter0 + 1

    Y= np.dot(A,np.transpose(B))
    Y_recover = Y

    return Y_recover

#矩阵分解算法
def run_MC(Y):
    maxiter = 1000
    alpha = 0.0001
    #SVD
    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    r = 50
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    U= U[:, 0:r]
    V= V[0:r,:]
    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    C=Wt
    L  = TCMF1(alpha, Y, maxiter,A,B,C)
    return L





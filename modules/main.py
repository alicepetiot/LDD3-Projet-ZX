import numpy as np 
import itertools
from itertools import product

"""
    @name : 
        - print_list
    @parameters :
        - @list [list] : a list which contains 2D numpy array 
    @return : 
        - @_ [void] : no return
    @explanation : 
        - print each elements of the list 
"""
def print_list(list):
    print("[")
    for arr in list: print(arr)
    print("]")

"""
    @name : 
        - Q1
    @parameters :
        - @r [int] : size of the Q matrix
    @return : 
        - @res [list] : a list which contains 2D numpy array 
    @explanation : 
        - generates all the possible Q matrix when r equal n 
"""
def Q1(r):
    res, a, cpt = [], 0, 0 
    while(cpt < 2):
        Q = np.zeros((r,r), dtype=int) if cpt == 0 else np.ones((r,r), dtype=int)
        for i in range(r): Q[i][i] = a
        if a == 3: 
            cpt += 1
            a = -1
        a += 1
        res.append(Q)
    return res

"""
    @name : 
        - Q2
    @parameters :
        - @r [int] : size of the Q matrix
    @return : 
        - @res [list] : a list which contains 2D numpy array 
    @explanation : 
        - generates all the possible Q matrix when r equal n+1
"""
def Q2(r):
    res, a, cpt = [], 0, 0
    while(cpt < 2):
        Q_r_equal_n = np.zeros((r,r), dtype=int) if cpt == 0 else np.ones((r,r), dtype=int)
        for i in range(r): Q_r_equal_n[i][i] = a
        Q_plus_column = np.insert(Q_r_equal_n, 0, [1]*r,axis=1)
        for i in range(2):
            Q = np.insert(Q_plus_column, 0, [1]*(r+1),axis=0)
            Q[0,0] = i
            res.append(Q)
        if a == 3: 
            cpt += 1
            a = -1
        a += 1
    return res

"""
    @name : 
        - Q3
    @parameters :
        - @r [int] : size of the Q matrix
    @return : 
        - @res [list] : a list which contains 2D numpy array 
    @explanation : 
        - generates all the possible Q matrix when r < n 
"""
def Q3(r):
    res,cpt = [],0
    sym = list(itertools.product([0,1],repeat=r))
    diag = list(itertools.product([0,1,2,3],repeat=int(((r*r)-r)/2)))       
    for e in sym:
        for d in diag:
            for i in range(r):
                for l in range(1,r):
                    for c in range(l+1):
                        M = np.zeros([r,r],dtype=int)
                        M[i,i] = d[i]
                        M[l,c] = e[i] 
                        M[c,l] = e[i]
            res.append(M)
    return res

"""
    @name : 
        - generate_matrix
    @parameters :
        - r [int] : size of elements 
        - n [int] : numbers of qubits
    @return : 
        - Q [list] : a list which contains 2D numpy array 
        - A [list] : a list which contains 2D numpy array 
        - B [list] : a list which contains 2D numpy array 
    @explanation : 
        - Q : contains all the possibles matrix Q in function of the r and n values 
        - A : contains all the possibles matrix A in function of the r and n values
        - B : contains all the possibles matrix B in function of the r and n values
"""
def generate_matrix(r,n):
    B = [np.zeros([n,1], dtype=int),np.ones([n,1], dtype=int)]
    if r == n:
        A = [np.eye(r, dtype=int)]
        Q = Q1(r)
    elif r == n+1: 
        A = [np.c_[np.zeros((n,), dtype=int),np.eye(n, dtype=int)]]
        Q = Q2(r)
    else: 
        A = [np.ones((n,r), dtype=int)] #n lignes, r colonnes 
        Q = [[]]
    return Q,A,B

#res = generate_matrix(4,3)
#print_list(res[0])
#print_list(res[1])
#print_list(res[2])

print_list(Q3(3))
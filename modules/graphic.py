import numpy as np 
from .matrix import transpose
import math 

# variables 
ket0 = np.array([[1],[0]])
ket1 = np.array([[0],[1]])
bra0 = [np.array([1,0])]
bra1 = [np.array([0,1])]
H = 1/math.sqrt(2) * np.array([[1,1],[1,-1]])
ketp = H.dot(ket0)
ketm = H.dot(ket1)
brap = transpose(ketp)
bram = transpose(ketm)

def tensor_product(m1,m2):
    res = []
    s = np.shape(m1) 
    s2 = np.shape(m2)
    
    if s == (0,) or s2 == (0,) or s == (1,0) or s2 == (1,0):
        return []
    
    if len(s) == 1 and len(s2) != 1:
        for k in range(len(m2)):
            line = []
            for i in range(len(m1)):
                line = line + list(m1[i]*m2[k])
            res.append(line)
    elif len(s2) == 1 and len(s) != 1:
        for i in range(len(m1)):
            line = []
            for j in range(len(m1[0])):
                line = line + list(m1[i][j] * m2)
            res.append(line)
    elif len(s) == 1 and len(s2) == 1:
        line = []
        for i in range(len(m1)):
            line = line + list(m1[i]*m2)
        res.append(line)
    else:         
        for i in range(len(m1)):
            for k in range(len(m2)):
                line = [] 
                for j in range(len(m1[0])):
                    line = line + list(m1[i][j] * m2[k])
                res.append(line)
    return np.array(res)

def tensor_product_pow(m,nb):
    res = m
    if nb == 0:
        res = []
    else:
        for i in range(nb-1):
            res = tensor_product(res,m)
    return res

def green_node(n,m,a):
    mat = np.zeros([2**m,2**n],dtype=complex) 
    if n == 0 and m == 0 : 
        mat[0,0] = 1+math.cos(a) + 1j*math.sin(a)
    else:
        mat[0,0] = 1
        mat[len(mat)-1,len(mat[0])-1] = math.cos(a) + 1j*math.sin(a)
    return mat 

def gn(n,m,a):
    x1 = tensor_product_pow(ket0,m)
    x2 = tensor_product_pow(ket1,m)
    x3 = tensor_product_pow(bra0,n)
    x4 = tensor_product_pow(bra1,n)
    m1 = x1.dot(x3)
    m2 = (math.cos(a) + 1j*math.sin(a)) * x2.dot(x4)
    return np.add(m1,m2)

def rn(n,m,a):
    x1 = tensor_product_pow(ketp,m)
    x2 = tensor_product_pow(ketm,m)
    x3 = tensor_product_pow(brap,n)
    x4 = tensor_product_pow(bram,n)
    m1 = x1.dot(x3)
    m2 = (math.cos(a) + 1j*math.sin(a)) * x2.dot(x4)
    return np.add(m1,m2)

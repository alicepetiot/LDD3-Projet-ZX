import numpy as np
import itertools

def Q_r_equal_n(r):
    
    res, a, b = [], 0, 0 
    
    if isinstance(r,bool):
        raise Exception("The parameter r must be an integer")  
    if not isinstance(r,int):
        raise Exception("The parameter r must be an integer")  
    if r <= 0:
        raise Exception("The parameter r must be strictly greater than zero")
    
    if r == 1:
        res = [[[0]],[[1]],[[2]],[[3]]]
    else: 
        while(b < 2):
            Q = np.zeros((r,r), dtype=int) if b == 0 else np.ones((r,r), dtype=int)
            for i in range(r): 
                Q[i][i] = a
            if a == 3: 
                b += 1
                a = -1
            a += 1
            res.append(Q)
    
    return res

def Q_r_equal_n_plus_1(r):
    
    res, a, b = [], 0, 0 

    if (isinstance(r,bool)) or (not isinstance(r,int)):
        raise Exception("The parameter r must be an integer")  
    if r <= 0:
        raise Exception("The parameter r must be strictly greater than zero")
    
    if r == 1:
        res = [[[0]],[[1]],[[2]],[[3]]]
    elif r == 2:
        res = [
                    np.array([[0,1],[1,0]]),
                    np.array([[1,1],[1,0]]),
                    np.array([[0,1],[1,1]]),
                    np.array([[1,1],[1,1]]),
                    np.array([[0,1],[1,2]]),
                    np.array([[1,1],[1,2]]),
                    np.array([[0,1],[1,3]]),
                    np.array([[1,1],[1,3]])
                ]
    else:
        while(b < 2):
            Q1 = np.zeros((r-1,r-1), dtype=int) if b == 0 else np.ones((r-1,r-1), dtype=int) 
            for i in range(r-1): 
                Q1[i][i] = a
            Q2 = np.insert(Q1, 0, 1,axis=1)
            for i in range(2):
                Q = np.insert(Q2, 0, 1,axis=0)
                Q[0,0] = i
                res.append(Q)
            if a == 3: 
                b += 1
                a = -1
            a += 1

    return res

def Q_r_less_than_n(r):
    res = []
 
    if (isinstance(r,bool)) or (not isinstance(r,int)):
        raise Exception("The parameter r must be an integer")  
    if r <= 0:
        raise Exception("The parameter r must be strictly greater than zero")
    
    diag = list(itertools.product([0,1,2,3], repeat=r))
    sym = list(itertools.product([0,1],repeat=int(((r*r)-r)/2)))  
    for a in diag: 
        for b in sym:
            M = np.zeros([r,r],dtype=int) 
            for i in range(r):
                M[i][i] = a[i]
            if r == 2:                                
                M[1,0] = b[0]
                M[0,1] = b[0]
            else:
                cpt = 0
                for l in range(1,r): 
                    for c in range(l): 
                        M[l,c] = b[cpt]
                        M[c,l] = b[cpt]
                        cpt+=1
            res.append(M)  

    return res

def generate_matrix(r,n):
    
    if (isinstance(r,bool) or isinstance(n,bool)) or (not isinstance(r,int) or not isinstance(n,int)):
        raise Exception("r and n must be integers")  
    if r <= 0 or n <= 0:
        raise Exception("r and n must be strictly greater than zero")
    if r >= n+2:
        raise Exception("r must be strictly lower than n+2")
    
    if r == n:
        A = np.eye(r, dtype=int)
        B = [np.zeros([n,1], dtype=int),np.ones([n,1], dtype=int)]
        Q = Q_r_equal_n(r)
    elif r == n+1: 
        A = np.c_[np.zeros((n,), dtype=int),np.eye(n, dtype=int)]
        B = [np.zeros([r,1], dtype=int),np.ones([r,1], dtype=int)]
        Q = Q_r_equal_n_plus_1(r)
    else: 
        A = np.ones((n,r), dtype=int)
        B = [np.zeros([n,1], dtype=int),np.ones([n,1], dtype=int)]
        Q = Q_r_less_than_n(r)
        
    return Q,A,B

def transpose(m):
    s = np.shape(m) 
    l = len(m) 
    
    if l == 0:
        res = [[]]
    elif len(s) == 1: 
        res = np.zeros((s[0],1)) 
        for i in range(l):
            res[i] = m[i] 
    else: 
        res = [] 
        for j in range(len(m[0]) ): 
            temp = []
            for i in range(l): 
                temp.append(m[i][j])
            res.append(temp) 
    return np.array(res)
    
def cartesian_product(arr1,arr2): 
    if (
        len(arr1) == 0 or 
        len(arr2) == 0 or
        np.shape(arr1) == (1,0) or
        np.shape(arr2) == (1,0)
        ):
        return []
    else: 
        res = []
        s1 = np.shape(arr1[0])
        s2 = np.shape(arr2[0])
        if s1 == () and s2 == ():
            for i in range(len(arr1)):
                for j in range(len(arr2)):
                    arr = np.array([[arr1[i]],[arr2[j]]])
                    res.append(arr)    
        elif s1 == () and not s2 == ():
            for k in range(len(arr1)):
                for i in range(len(arr2)):
                    arr = [[arr1[k]]]
                    for j in range(len(arr2[0])):
                        arr.append([arr2[i][j]])
                    res.append(arr)
        elif not s1 == () and s2 == ():
            for i in range(len(arr1)):
                for k in range(len(arr2)):
                    arr = []
                    for j in range(len(arr1[0])):
                        arr.append([arr1[i][j]])
                        if j == len(arr1[0])-1:
                            arr.append([arr2[k]])
                    res.append(arr)
        elif not s1 == () and not s2 == ():
            for i in range(len(arr1)):
                for k in range(len(arr2)):
                    arr = []
                    for j in range(len(arr1[0])):
                        arr.append([arr1[i][j]])
                    for l in range(len(arr2[0])):
                        arr.append([arr2[k][l]])
                    res.append(arr)
        return res

def cartesian_product_repeat(arr,n):
    
    if n <= 0: 
        raise Exception("n must be strictly greater than zero")
    
    if n == 1:
        res = []
        for v in arr:
            res.append([[v]])
        return res 
    else:
        res = arr
        for i in range(n-1):
            res = cartesian_product(res,arr)
        return res

def xor_matrix(m1,m2):
    sh1,sh2,res = np.shape(m1),np.shape(m2),[]

    if sh1 != sh2 or sh1 == (0,) or sh1 == (1,0):
        raise Exception("The lenght of m1 and m2 must be equal")

    if sh1 == (sh1[0],):
        for i in range(len(m1)):
            if m1[i] == m2[i]:
                res.append(0) 
            else: 
                res.append(1)
            
    else:
        for i in range(len(m1)):
            line = []
            for j in range(len(m1[0])):
                if m1[i][j] == m2[i][j]:
                    line.append(0) 
                else: 
                    line.append(1)
            res.append(line)
            
    return np.array(res)

def arr_to_tuple(arr):
    sh = np.shape(arr)
    if sh == (sh[0],):
        lines = []
        for i in range(len(arr)):
            val = arr[i]
            lines.append((int(val.real),int(val.imag)))
        return tuple(lines)
    else:
        liste = []
        for i in range(len(arr)):
            lines = []
            for j in range(len(arr[0])):
                value = arr[i][j]
                lines.append((int(value.real),int(value.imag)))
            liste.append(tuple(lines))
        return tuple(liste)
    
def tuple_to_arr(tpl):
    sh = np.shape(tpl)
    
    if len(tpl) < 0 or len(tpl) > 3:
        raise Exception("The tuple must be of shape (_,); (_,2); (_,_,2)")
    
    if len(sh) == 1:
        return np.array([])
    elif sh == (1,0):
        return np.array([[]])
    elif len(sh) == 2:
        res = []
        for cplx in tpl:
            res.append(cplx[0]+cplx[1]*1j)
        return np.array(res)
    elif len(sh) == 3:
        res = []
        for line in tpl:
            res_line = []
            for cplx in line:
                res_line.append(cplx[0]+cplx[1]*1j)
            res.append(res_line)
        return np.array(res)
    
def set_to_list(set):
    res = []
    for val in set: 
        res.append(tuple_to_arr(val)) 
    return res 
    
def sum_matrix(n_max):
    v = set()
    for n in range(1,n_max+1):
        for r in range(1,n+1):
            xr = cartesian_product_repeat(np.array([0,1]),r)
            tmp = generate_matrix(r,n)
            A = tmp[1]
            B = tmp[2]
            Q = tmp[0]
            for b in range(len(B)):
                for q in range(len(Q)):
                    res = None
                    for x in range(len(xr)):
                        imaginary_phase = 1j**(transpose(xr[x]).dot(Q[q])).dot(xr[x])
                        matrix_in_ket = xor_matrix(A.dot(xr[x]),B[b])
                        nb_in_matrix_in_ket = int("".join(map(str,transpose(matrix_in_ket)[0])),2)
                        standard_basis_offset = np.zeros([2**n])
                        standard_basis_offset[nb_in_matrix_in_ket] = 1
                        tmp_res = imaginary_phase*standard_basis_offset
                        if res is None:
                            res = np.zeros(np.shape(tmp_res),dtype=object)
                        res = np.add(res,tmp_res)
                    v.add(arr_to_tuple(res))
    #print("V IN SUM MATRIX:",v)
    #print(len(v))
    return v


#iterator = iter(my_set)
#item1 = next(iterator, None)
#print(item1)
#item2 = next(iterator, None)
#print(item2)
#tuple_to_arr(item1)

import numpy as np 
from hashlib import sha1
import re

def save_matrix(liste,path):
    with open(path,'w') as f:
        for mat in liste:
            s = np.shape(mat) 
            l = len(mat)
            f.write(sha1(mat).hexdigest()+"\n")
            if len(s) == 1:
                for i in range(l):
                    c = mat[i] 
                    real = c.real 
                    imag = c.imag 
                    f.write('('+str(real)+','+str(imag)+')'+' ')
                f.write("\n")
            else:
                for i in range(l): 
                    for j in range(len(mat[0])): 
                        c = mat[i,j]
                        real = c.real
                        imag = c.imag 
                        f.write('('+str(real)+','+str(imag)+')'+' ')
                    f.write("\n")
                f.write("\n")
        f.write("END")
    
def read_matrix(path):
    res = []
    mat = []
    with open(path,'r') as f:
        for line in f:    
            if line.startswith("\n") or line.startswith("END"):
                res.append(mat)
                mat = []      
            elif not line.startswith("(") :
                pass
            else:
                split = line.split(" ")
                l = []
                for i in range(len(split)-1):
                    real = re.findall('[0-9][.]*[0-9]*',split[i])[0]
                    imag = re.findall('[0-9][.]*[0-9]*',split[i][1+len(real)+1:])[0]
                    l.append(complex(int(float(real)),int(float(imag))))
                mat.append(l)
    return np.array([np.array(x) for x in res],dtype=object)

def search_matrix_from_hash(path,h):
    with open(path,'r') as f:
        boolean = False
        for line in f:  
            if not line.startswith("(") and not line.startswith("END") and not line.startswith("\n"):
                if line.split()[0] == h: 
                    boolean = boolean or True
                else: 
                    boolean = boolean or False  
            else:
                pass 
    return boolean
                
def search_matrix(l,h):
    boolean = False
    for m in l:
        if sha1(m).hexdigest() == h:
            boolean = True 
            break 
    return boolean

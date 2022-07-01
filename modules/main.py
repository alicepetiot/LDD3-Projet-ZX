import numpy as np 
from hashlib import sha1
import itertools
import math
import re 

#fonction pour afficher une simple matrice numpy
def print_simp_arr(arr):
    print("\n----------\n[", end="")
    for i in range(len(arr)):
        print(arr[i], end="") if i == len(arr) - 1 else print(arr[i], end=" ")
    print("]\n----------")

#fonction pour afficher une double matrice numpy 
def print_arr(arr):
    print("\n----------\n[", end="")
    for i in range(len(arr)):
        print("[", end="")
        for j in range(len(arr[0])):
            print(arr[i][j], end="") if j == len(arr[0]) - 1 else print(arr[i][j], end=" ")
        print("]",end="") if i == len(arr) - 1 else print("]")
    print("]\n----------")

#fonction pour afficher une liste de matrice numpy 
def print_list(list):
    print("\n[", end="")
    for arr in list: 
        print_arr(arr)
    print("]\n") 

def Q1(r):
    # on stocke dans "res" toutes les matrices qu'on va générer 
    # on stocke dans "a" la valeur de la diagonale (0 ou 1 ou 2 ou 3)
    # on stocke dans "b" la valeur des autres cases (0 ou 1)
    res, a, b = [], 0, 0 
    # si r vaut 1 alors c'est un cas special (on aura aucune valeur de b)
    if r == 1:
        res = [[[0]],[[1]],[[2]],[[3]]]
    else: 
        # on verifie bien que la taille donnee pour la matrice est positive 
        if (r > 1):
            # ici on va commencer a creer toutes les matrices pour lesquels "b" vaut 0 
            # et seulement changer la valeur de la diagonale a chaque iteration
            # on change de "b" seulement quand on a fait toutes les valeurs de "a"
            # cad lorsque "a" vaut 3
            while(b < 2):
                # si "b" vaut 0 alors on init la matrice courante de taille (r*r) avec des zeros
                # sinon on initialize Q avec des uns
                Q = np.zeros((r,r), dtype=int) if b == 0 else np.ones((r,r), dtype=int)
                # on commence par remplir la diagonale avec la valeur courante de "a"
                for i in range(r): 
                    Q[i][i] = a
                # si "a" vaut 3 alors on incremente b et on oublie pas de remettre "a" a 0
                if a == 3: 
                    b += 1
                    a = -1
                # sinon on continue en changeant la valeur de a pour la prochaine matrice 
                # cad en augmentant sa valeur de 1
                a += 1
                # finalement on ajoute la matrice a "res"
                res.append(Q)
        # si r est negatif alors on print une erreur 
        else: 
            print("Error => r :",r,"doit etre strictement superieur a 0")
    return res

def Q2(r):
    # on stocke dans res toutes les matrices qu'on va générer 
    # on stocke dans a la valeur que va prendre la diagonale au fur et a mesure 
    # on stocke dans b la valeur que vont prendre toutes les autres cases 
    res, a, b = [], 0, 0 
    if (r > 0):
        while(b < 2):
            # initialement Q1 est une matrice de taille (r-1,r-1) qui est remplie de 0 
            # sa diagonale prendra que des 0, que des 1, que des 2 ou que des 3
            # toutes les autres cases prendront que des 0 ou que des 1
            Q1 = np.zeros((r-1,r-1), dtype=int) if b == 0 else np.ones((r-1,r-1), dtype=int) 
            # on commence par remplir la diagonale de la matrice Q1 avec la valeur a 
            for i in range(r-1): 
                Q1[i][i] = a
            # on insere dans Q1 une colonne avec que des 1 
            # Q2 a donc (r-1) lignes et r colonnes 
            Q2 = np.insert(Q1, 0, 1,axis=1)
            for i in range(2):
                # ici on insere dans Q2 une ligne avec que des 1 
                # Q est donc une matrice carree de taille r*r 
                Q = np.insert(Q2, 0, 1,axis=0)
                # finalement on va changer la valeur en (0,0) en 0 ou 1 qui correspond a la valeur de i
                Q[0,0] = i
                # puis on ajoute la matrice qu'on vient de creer a res 
                res.append(Q)
            # si on a finit de generer toutes les matrices avec b = 0 et a qui varie entre 0 et 3 
            # alors on incremente b pour generer toutes les autres matrices ou b = 1 avec a qui varie 
            # on oublie pas de mettre a jour a pour repartir de 0 (jusqu'a 3)
            if a == 3: 
                b += 1
                a = -1
            # si on a pas fini avec b alors on incremente a pour changer la valeur de la diagonale 
            a += 1
    else:
        print("Error => r :",r,"doit etre strictement superieur a 0")
    return res

def Q3(r):
    # res contient toutes les matrices qu'on va générer 
    # diag contient toutes les possibilites de la diagonale dans des tuples de tailles r 
    # sym contient toutes les possibilites du triangle inferieur de la matrice dans des tuples de taille (r^2 - r) / 2
    res = []
    if r >= 0:
        diag = list(itertools.product([0,1,2,3], repeat=r))
        sym = list(itertools.product([0,1],repeat=int(((r*r)-r)/2))) 
        # pour une diagonale "a" dans diag    
        for a in diag: 
            # et pour chaque disposition "b" dans sym 
            for b in sym:
                # on commence par initialiser une matrice de taille (r*r) avec des 0
                M = np.zeros([r,r],dtype=int) 
                # puis on remplit sa diagonale avec les valeurs dans le tuple "a" courant 
                for i in range(r):
                    M[i][i] = a[i]
                # si r vaut 2 alors on se trouve dans un cas particulier 
                # on remplit donc les cases (1,0) et (0,1) avec un 0 
                # puis pour la prochaine matrice M on fera la meme chose avec un 1
                if r == 2:                                
                    M[1,0] = b[0]
                    M[0,1] = b[0]
                # si r n est pas egal a 2 alors on a une formule pour remplir le reste de la matrice 
                else:
                    # la variable cpt permet d avancer au fur et a mesure dans le tuple courant "b"
                    cpt = 0
                    # "l" correspond aux lignes et "c" correspond aux colonnes du triangle inferieur de "M"
                    for l in range(1,r): 
                        for c in range(l): 
                            # si on a un tuple de la forme (0,0,1) pour r = 4 alors on aura M[1,0] = 0, M[2,0] = 0
                            # et M[2,1] = 1 avec M[l,c] donc l va de (1 à r-1) et c de (0 à l-1)
                            M[l,c] = b[cpt]
                            # puis une fois qu on a remplit ces cases il faut remplir leur symetrique 
                            M[c,l] = b[cpt]
                            # on augmente le cpt 
                            cpt+=1
                # finalement on ajoute la matrice "M" a "res"
                res.append(M)  
    else:
        print("Error => r :",r,"doit etre superieur a 0")
    return res

# permet de generer toutes les matrices Q, A et B en fonction des valeurs de r et n 
def generate_matrix(r,n):
    # peut importe les valeurs de r et n les matrices B sont toujours les mêmes 
    # B est une liste qui contient deux matrices avec 1 colonne et n lignes
    # la premiere est une colonne avec des zeros et la deuxieme avec des uns
    B = [np.zeros([n,1], dtype=int),np.ones([n,1], dtype=int)]
    # on differencie 3 cas : 
    # 1) r est egal a n 
    # 2) r est egal a n+1 
    # 3) r est plus petit que n 
    if r == n:
        # A est la matrice identite (r colonnes, n lignes)
        A = [np.eye(r, dtype=int)]
        # Q est une matrice symetrique (r colonnes, r lignes)
        # sa diagonale comporte que des 0 (ou que des 1,2 ou 3)
        # toutes ses autres valeurs sont soient egal a 0 ou alors a 1
        Q = Q1(r)
    elif r == n+1: 
        # A est une matrice identite de taille (r-1 colonnes, n-1 lignes)
        # avec une colonne qui contient que des zeros tout a gauche
        A = [np.c_[np.zeros((n,), dtype=int),np.eye(n, dtype=int)]]
        # Q est toujours une matrice symetrique 
        # on retrouve la meme forme que Q1 de taille (r-1 colonnes, r-1 lignes)
        # et en plus on a une colonne a droite avec que des 1 
        # une lignes en haut qu'avec des 1 
        # et enfin la case (0,0) peut prendre un 0 ou un 1 
        Q = Q2(r)
    else: 
        # A est une matrice complète avec (r colonnes, n lignes) 
        A = [np.ones((n,r), dtype=int)] 
        # Q est toujours symétrique mais quelconque cette fois-ci 
        Q = Q3(r)
    return Q,A,B

# permet de transposer une matrice double ou simple 
def transpose(m):
    s = np.shape(m) # shape = (x,y) avec x les lignes et y les colonnes
    l = len(m) # nb lignes 
    if len(s) == 1: # si la shape est de la forme (x,_) alors len(s) = 1 et on a une matrice simple
        res = np.zeros((s[0],1)) # on prepare notre matrice pour etre sous la forme d une colonne 
        for i in range(l):
            res[l-i-1] = m[i] # on la remplit 
    # sinon c est qu on a des matrices doubles 
    else: 
        res = [] 
        for j in range(len(m[0]) ): # on parcours une colonne 
            temp = []
            for i in range(l): # on recupere ses valeurs pour chaque lignes 
                temp.append(m[i][j])
            res.append(temp) # on ajoute ses valeurs a notre nouvelle matrice mais en tant que ligne 
        # on retransforme notre res en numpy array
        return np.array(res)

# retourne le produit tensoriel entre deux matrices  
def tensor_product(m1,m2):
    res = []
    # on regarde une lignes dans m1 
    # on regarde le premier membre de la ligne i et on le multiplie par la premiere ligne dans m2 (on obtient une liste)
    # puis on regarde le deuxieme membre de la ligne i et on le multiplie par la premiere ligne dans m2 (on obtient une liste)
    # on fait ca pour chaque membre de la ligne et on concatene tous les resultats dans line 
    # ca nous donne la premiere ligne de notre matrice 
    # puis on recommence pour la meme ligne i mais avec les lignes suivantes k dans m2 
    # lorsqu on a fini de multiplier i avec toutes les lignes k dans m2, on change de lignes i 
    # i represente les lignes de m1
    for i in range(len(m1)):
        # k represente les lignes de m2
        for k in range(len(m2)):
            line = [] 
            # j represente les colonnes de m1
            for j in range(len(m1[0])):
                line = line + list(m1[i][j] * m2[k])
            res.append(line)
    # m1 =  0 0        m2 = 1 1         res =   0 0 0 0     =   [0 0] + [0 0]   (i = 0, k = 0, j = 0,1)
    #       0 2             1 1                 0 0 0 0         [0 0] + [0 0]   (i = 0, k = 1, j = 0,1)
    #       0 0                                 0 0 2 2         0 0 2 2         (i = 1, k = 0, j = 0,1)
    #                                           0 0 2 2         0 0 2 2         (i = 1, k = 1, j = 0,1)
    #                                           0 0 0 0         0 0 0 0         (i = 2, k = 0, j = 0,1)
    #                                           0 0 0 0         0 0 0 0         (i = 2, k = 1, j = 0,1)
    return np.array(res)

# permet de faire le produit tensoriel entre plusieurs matrices 
# prend une liste de matrice en parametres 
def tensor_products(list):
    res = []
    if len(list) >= 2:
        res = tensor_product(list[0],list[1])
        temp = list.copy()
        temp.pop(0)
        temp.pop(1)
        while len(temp) != 0:
            res = tensor_product(res,temp[0])
            temp.pop(0)
    else:
        print("Error, the list must contains at least 2 elements")
    return res

# produit tensoriel d une matrice avec elle meme nb fois 
def tensor_product_pow(m,nb):
    res = m
    if nb == 0:
        res = []
    else:
        for i in range(nb-1):
            res = tensor_product(res,m)
    return res

# variables 
ket0 = np.array([[1],[0]])
ket1 = np.array([[0],[1]])
bra0 = np.array([1,0])
bra1 = np.array([0,1])
H = 1/math.sqrt(2) * np.array([[1,1],[1,-1]])
ketp = H.dot(ket0)
ketm = H.dot(ket1)
brap = transpose(ketp)
bram = transpose(ketm)

# genere un noeud vert avec n entrees, m sorties et avec un angle a 
def green_node(n,m,a):
    mat = np.zeros([2**m,2**n],dtype=complex) # matrice de type complexe 
    if n == 0 and m == 0 : 
        mat[0,0] = 1+math.cos(a) + 1j*math.sin(a)
    else:
        mat[0,0] = 1
        mat[len(mat)-1,len(mat[0])-1] = math.cos(a) + 1j*math.sin(a)
    return mat 

# genere un noeud vert avec n entrees, m sorties et avec un angle a 
def gn(n,m,a):
    x1 = tensor_product_pow([ket0],m)
    x2 = tensor_product_pow([ket1],m)
    x3 = tensor_product_pow([bra0],n)
    x4 = tensor_product_pow([bra1],n)
    m1 = x1.dot(x3)
    m2 = (math.cos(a) + 1j*math.sin(a)) * x2.dot(x4)
    return np.add(m1,m2)

# genere un noeud rouge avec n entrees, m sorties et avec un angle a 
def rn(n,m,a):
    x1 = tensor_product_pow([ketp],m)
    x2 = tensor_product_pow([ketm],m)
    x3 = tensor_product_pow([brap],n)
    x4 = tensor_product_pow([bram],n)
    m1 = x1.dot(x3)
    m2 = (math.cos(a) + 1j*math.sin(a)) * x2.dot(x4)
    return np.add(m1,m2)

# permet de generer le hash d une matrice pour la reconnaitre 
def hash_mat(arr):
    return sha1(arr).hexdigest()

# permet d ecrire des matrices complexes dans un fichier txt (1D ou 2D)
def save_matrix(liste,path):
    with open(path,'w') as f:
        for mat in liste:
            s = np.shape(mat) # shape = (x,y) avec x les lignes et y les colonnes
            l = len(mat)
            f.write(hash_mat(mat)+"\n")
            if len(s) == 1: # si la shape est de la forme (x,_) alors len(s) = 1 et on a une matrice simple
                for i in range(l):
                    c = mat[i] # le complexe courrant
                    real = c.real # sa partie reelle
                    imag = c.imag # sa partie imaginaire 
                    f.write('('+str(real)+','+str(imag)+')'+' ')
                f.write("\n")
            else:
                for i in range(l): #lignes
                    for j in range(len(mat[0])): #colonnes
                        c = mat[i,j]
                        real = c.real
                        imag = c.imag 
                        f.write('('+str(real)+','+str(imag)+')'+' ')
                    f.write("\n")
                f.write("\n")
        f.write("END")


#liste = [green_node(1,2,0)]*10+[np.array([0,2])]
#save_matrix(liste,"matrix/test.txt")
    
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
                #print("cc")
                split = line.split(" ")
                l = []
                for i in range(len(split)-1):
                    real = re.findall('[0-9][.]*[0-9]*',split[i])[0]
                    imag = re.findall('[0-9][.]*[0-9]*',split[i][1+len(real)+1:])[0]
                    l.append(float(real)+float(imag)*1j)
                #print("l:", l)
                mat.append(l)
                #print("mat:",mat)
    return res


res = read_matrix("matrix/test.txt")
print_list(res)
                

def sum_matrix(r,n):
    i = 1j 
    all_x = np.array(list(itertools.product([0,1],repeat=r)))
    x = all_x[0]
    #print(np.shape(x))
    transpose(x)
    #xt = transpose(x)
    res = generate_matrix(r,n)
    Q = res[0][1]
    A = res[1][0]
    B = res[2][0]
    #print("x:",np.shape(x))
    #print(np.shape(xt))
    #print_arr(xt)
    #print("Q:",Q)
    #print("A:",A)
    #print_arr(B)
    #print("1:",i**(xt.dot(Q)).dot(x))
    #print("2:",A.dot(x).T)
    #print("3:",np.bitwise_xor(A.dot(x),B))
    return []

#sum_matrix(2,3)

 
 
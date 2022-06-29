import numpy as np 
import itertools
from itertools import product

def print_list(list):
    print("[")
    for arr in list: print(arr)
    print("]")


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


""" 
    @ name : 
        - generate_matrix
    @ parameters : 
        - r [int+] : nombre de variables
        - n [int+] : nombre de qubits 
    @ returns : 
        - Q [list] : liste de matrices dont la forme dépend des valeurs r et n 
        - A [list] : liste de matrices dont la forme dépend des valeurs r et n 
        - B [list] : liste de matrices dont la forme dépend des valeurs r et n 
    @ goal : 
        - generer 3 listes Q, A et b qui contiennent toutes les matrices possibles 
        qui leur correspond en fonction des valeurs de r et n 
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
        Q = Q3(r)
    return Q,A,B


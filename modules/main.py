from matrix import *
from files import *

n = 2
path = "bank/zx_matrix.txt"

my_set = sum_matrix(n)
my_bank = set_to_list(my_set)
save_matrix(my_bank,path)



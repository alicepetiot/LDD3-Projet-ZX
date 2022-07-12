import sys
sys.path.append('../') # allows us to fetch files from the project root

import unittest
from modules.files import *
from modules.graphic import gn

class MatrixTest(unittest.TestCase):
    
    def setUp(self):
        self.path = "tests/bank_test/zx_matrix_test.txt"
        self.gn = gn(3,2,np.pi/2)
        self.arr = np.array([0,1j])
        self.liste = np.array([self.gn]*10 + [self.arr],dtype=object)
        self.hash0 = sha1(self.gn).hexdigest()
        self.hash1 = sha1(self.arr).hexdigest()
        self.hash2 = sha1(np.array([0,0])).hexdigest()
    
    def test_save_matrix(self): 
        save_matrix(self.liste,self.path)
        
        print("test_save_matrix CHECK")
    
    def test_read_matrix(self): 
        res = read_matrix(self.path)
        #print(res)
        print("test_read_matrix CHECK")
    
    def test_search_matrix_from_hash(self): 
        res0 = search_matrix_from_hash(self.path,self.hash0)
        res1 = search_matrix_from_hash(self.path,self.hash1)
        res2 = search_matrix_from_hash(self.path,self.hash2)
        
        self.assertTrue(res0)
        self.assertTrue(res1)
        self.assertFalse(res2)
        
        print("test_search_matrix_from_hash CHECK")
        
    def test_search_matrix(self):
        
        res0 = search_matrix(self.liste,self.hash0)
        res1 = search_matrix(self.liste,self.hash1)
        res2 = search_matrix(self.liste,self.hash2)
        
        self.assertTrue(res0)
        self.assertTrue(res1)
        self.assertFalse(res2)
        
        print("test_search_matrix CHECK")
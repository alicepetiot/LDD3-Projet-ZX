import sys
sys.path.append('../') # allows us to fetch files from the project root

import unittest
from modules.matrix import *

class MatrixTest(unittest.TestCase):    
    
    def test_Q_r_equal_n(self):
        
        with self.assertRaises(Exception) as context:
            Q_r_equal_n('petit test')
            
        with self.assertRaises(Exception) as context:
            Q_r_equal_n(True)
        
        with self.assertRaises(Exception) as context:
            Q_r_equal_n(-2)
        
        with self.assertRaises(Exception) as context:
            Q_r_equal_n(0)
            
        res0 = Q_r_equal_n(1)
        res1 = Q_r_equal_n(2)
        res2 = Q_r_equal_n(6)
        res3 = Q_r_equal_n(10)
        
        ega0 = [[[0]],[[1]],[[2]],[[3]]]
        
        ega1 = [    
                    np.array([[0,0],[0,0]]),
                    np.array([[1,0],[0,1]]),
                    np.array([[2,0],[0,2]]),
                    np.array([[3,0],[0,3]]),
                    np.array([[0,1],[1,0]]),
                    np.array([[1,1],[1,1]]),
                    np.array([[2,1],[1,2]]),
                    np.array([[3,1],[1,3]])
                ]
    
        self.assertEqual(len(res0),4)
        self.assertEqual(len(res1),8)
        self.assertEqual(len(res2),8)
        self.assertNotEqual(len(res3),5)
        
        self.assertEqual(res0,ega0)
        self.assertTrue(np.array_equal(res1,ega1))
        
        print("test_Q_r_equal_n CHECK")
    
    def test_Q_r_equal_n_plus_1(self):
        
        with self.assertRaises(Exception) as context:
            Q_r_equal_n_plus_1(0.111)
            
        with self.assertRaises(Exception) as context:
            Q_r_equal_n_plus_1(True)
        
        with self.assertRaises(Exception) as context:
            Q_r_equal_n_plus_1(-2)
        
        with self.assertRaises(Exception) as context:
            Q_r_equal_n_plus_1(0)
            
        res0 = Q_r_equal_n_plus_1(1)
        res1 = Q_r_equal_n_plus_1(2)
        res2 = Q_r_equal_n_plus_1(3)
        res3 = Q_r_equal_n_plus_1(6)
        
        ega0 = [[[0]],[[1]],[[2]],[[3]]]
        
        ega1 = [
                    np.array([[0,1],[1,0]]),
                    np.array([[1,1],[1,0]]),
                    np.array([[0,1],[1,1]]),
                    np.array([[1,1],[1,1]]),
                    np.array([[0,1],[1,2]]),
                    np.array([[1,1],[1,2]]),
                    np.array([[0,1],[1,3]]),
                    np.array([[1,1],[1,3]])
                ]
        
        ega2 =  [   
                    np.array([[0,1,1],[1,0,0],[1,0,0]]),
                    np.array([[1,1,1],[1,0,0],[1,0,0]]),
                    np.array([[0,1,1],[1,1,0],[1,0,1]]),
                    np.array([[1,1,1],[1,1,0],[1,0,1]]),
                    np.array([[0,1,1],[1,2,0],[1,0,2]]),
                    np.array([[1,1,1],[1,2,0],[1,0,2]]),
                    np.array([[0,1,1],[1,3,0],[1,0,3]]),
                    np.array([[1,1,1],[1,3,0],[1,0,3]]),
                    np.array([[0,1,1],[1,0,1],[1,1,0]]),
                    np.array([[1,1,1],[1,0,1],[1,1,0]]),
                    np.array([[0,1,1],[1,1,1],[1,1,1]]),
                    np.array([[1,1,1],[1,1,1],[1,1,1]]),
                    np.array([[0,1,1],[1,2,1],[1,1,2]]),
                    np.array([[1,1,1],[1,2,1],[1,1,2]]),
                    np.array([[0,1,1],[1,3,1],[1,1,3]]),
                    np.array([[1,1,1],[1,3,1],[1,1,3]])
            
                ]
        
        self.assertEqual(len(res0),4)
        self.assertEqual(len(res1),8)
        self.assertEqual(len(res2),16)
        self.assertEqual(len(res3),16)
        
        self.assertEqual(res0,ega0)
        self.assertTrue(np.array_equal(res1,ega1))
        self.assertTrue(np.array_equal(res2,ega2))
        
        print("test_Q_r_equal_n_plus_1 CHECK")
    
    def test_Q_r_less_than_n(self):
        
        with self.assertRaises(Exception) as context:
            Q_r_less_than_n("petit test")
            
        with self.assertRaises(Exception) as context:
            Q_r_less_than_n(True)
            
        with self.assertRaises(Exception) as context:
            Q_r_less_than_n(0.111)
        
        with self.assertRaises(Exception) as context:
            Q_r_less_than_n(-2)
        
        with self.assertRaises(Exception) as context:
            Q_r_less_than_n(0)
        
        
        """
            Formule : 4**r * 2**( r*(r-1)/2 ) = 2 ** ( 2r + r*(r-1)/2 )
        """
        res0 = Q_r_less_than_n(1)
        res1 = Q_r_less_than_n(2)
        res2 = Q_r_less_than_n(3)
        
        ega0 = [[[0]],[[1]],[[2]],[[3]]]
        
        ega1 = [
                    np.array([[0, 0],[0, 0]]), np.array([[0, 1],[1, 0]]), 
                    np.array([[0, 0],[0, 1]]), np.array([[0, 1],[1, 1]]),
                    np.array([[0, 0],[0, 2]]), np.array([[0, 1],[1, 2]]), 
                    np.array([[0, 0],[0, 3]]), np.array([[0, 1],[1, 3]]), 
                    np.array([[1, 0],[0, 0]]), np.array([[1, 1],[1, 0]]), 
                    np.array([[1, 0],[0, 1]]), np.array([[1, 1],[1, 1]]), 
                    np.array([[1, 0],[0, 2]]), np.array([[1, 1],[1, 2]]), 
                    np.array([[1, 0],[0, 3]]), np.array([[1, 1],[1, 3]]), 
                    np.array([[2, 0],[0, 0]]), np.array([[2, 1],[1, 0]]), 
                    np.array([[2, 0],[0, 1]]), np.array([[2, 1],[1, 1]]), 
                    np.array([[2, 0],[0, 2]]), np.array([[2, 1],[1, 2]]), 
                    np.array([[2, 0],[0, 3]]), np.array([[2, 1],[1, 3]]), 
                    np.array([[3, 0],[0, 0]]), np.array([[3, 1],[1, 0]]), 
                    np.array([[3, 0],[0, 1]]), np.array([[3, 1],[1, 1]]), 
                    np.array([[3, 0],[0, 2]]), np.array([[3, 1],[1, 2]]), 
                    np.array([[3, 0],[0, 3]]), np.array([[3, 1],[1, 3]])
                ]
        
        self.assertEqual(len(res0),4)
        self.assertEqual(len(res1),32)
        self.assertEqual(len(res2),512)
        
        self.assertEqual(res0,ega0)
        self.assertTrue(np.array_equal(res1,ega1))
        
        print("test_Q_r_less_than_n CHECK")
        
    
    def test_generate_matrix(self):
        
        with self.assertRaises(Exception) as context:
            generate_matrix(0,0) #r must be greater than 0
        
        with self.assertRaises(Exception) as context:
            generate_matrix(0,1) #r must be greater than 0
        
        with self.assertRaises(Exception) as context:
            generate_matrix(1,0) #n must be greater than 0
        
        with self.assertRaises(Exception) as context:
            generate_matrix(True,0) #r must be an integer
        
        with self.assertRaises(Exception) as context:
            generate_matrix(1,0.0) #n must be an integer
            
        with self.assertRaises(Exception) as context:
            generate_matrix(-2,2) #r must be greater than 0
            
        with self.assertRaises(Exception) as context:
            generate_matrix('aaaa',-2) #r must be an integer
        
        with self.assertRaises(Exception) as context:
            generate_matrix(3,1) #r must be strictly lower than n+2
        
        res0 = generate_matrix(1,1)
        res1 = generate_matrix(2,1)
        res2 = generate_matrix(1,2)
        res3 = generate_matrix(2,2)
        res4 = generate_matrix(3,4)
        
        A0,B0,Q0 = res0[1],res0[2],res0[0]
        A1,B1,Q1 = res1[1],res1[2],res1[0]
        A2,B2,Q2 = res2[1],res2[2],res2[0]
        A3,B3,Q3 = res3[1],res3[2],res3[0]
        A4,B4,Q4 = res4[1],res4[2],res4[0]
        
        egaA0 = np.array([[1]])
        egaA1 = np.array([[0,1]])
        egaA2 = np.array([[1],[1]])
        egaA3 = np.array([[1,0],[0,1]])
        egaA4 = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
        egaB0 = [np.array([[0]]),np.array([[1]])]
        egaB123 = [np.array([[0],[0]]),np.array([[1],[1]])]
        egaB4 = [np.array([[0],[0],[0],[0]]),np.array([[1],[1],[1],[1]])]
        
        self.assertTrue(np.array_equal(A0,egaA0))
        self.assertTrue(np.array_equal(B0,egaB0))
        self.assertEqual(len(Q0),4)
        
        self.assertTrue(np.array_equal(A1,egaA1))
        self.assertTrue(np.array_equal(B1,egaB123))
        self.assertEqual(len(Q1),8)
        
        self.assertTrue(np.array_equal(A2,egaA2))
        self.assertTrue(np.array_equal(B2,egaB123))
        self.assertEqual(len(Q2),4)
        
        self.assertTrue(np.array_equal(A3,egaA3))
        self.assertTrue(np.array_equal(B3,egaB123))
        self.assertEqual(len(Q3),8)
        
        self.assertTrue(np.array_equal(A4,egaA4))
        self.assertTrue(np.array_equal(B4,egaB4))
        self.assertEqual(len(Q4),512)
        
        print("test_generate_matrix CHECK")
    
    def test_transpose(self):
        
        res0 = transpose([])
        res1 = transpose([[]])
        res2 = transpose([0,1,2])
        res3 = transpose([[0,1,2]])
        res4 = transpose(np.array([[0,1],[2,3]]))

        self.assertTrue(np.array_equal(res0,[[]]))
        self.assertTrue(np.array_equal(res1,[]))
        self.assertTrue(np.array_equal(res2,[[0],[1],[2]]))
        self.assertTrue(np.array_equal(res3,[[0],[1],[2]]))
        self.assertTrue(np.array_equal(res4,[[0,2],[1,3]]))
        
        print("test_transpose CHECK")
    
    def test_cartesian_product(self):
        
        res0 = cartesian_product([],[])
        res1 = cartesian_product([[]],[[]])
        res2 = cartesian_product([],[1,2])
        res3 = cartesian_product([1,2],[])
        res4 = cartesian_product([],[[1,2],[1,2]])
        res5 = cartesian_product([[1,2],[1,2]],[])
        
        res6 = cartesian_product([0,1],[0,1])
        res7 = cartesian_product(   
                                    [0,1],
                                 
                                    [[0,0],
                                     [0,1],
                                     [1,0],
                                     [1,1]]
                                )
        res8 = cartesian_product(   
                                    [[0,0],
                                     [0,1],
                                     [1,0],
                                     [1,1]],
                                    
                                    [0,1]
                                )
        res9 = cartesian_product(   
                                    [[0,0],
                                     [0,1],
                                     [1,0],
                                     [1,1]],
                                    
                                    [[0,0],
                                     [0,1],
                                     [1,0],
                                     [1,1]]
                                )
        
        self.assertEqual(res0,[])
        self.assertEqual(res1,[])
        self.assertEqual(res2,[])
        self.assertEqual(res3,[])
        self.assertEqual(res4,[])
        self.assertEqual(res5,[])
        self.assertTrue(np.array_equal(res6,[
                                                [[0],[0]],
                                                [[0],[1]],
                                                [[1],[0]],
                                                [[1],[1]]]
                                       )
                        )
        self.assertTrue(np.array_equal(res7,                             
                                            [
                                                [[0],[0],[0]],
                                                [[0],[0],[1]],
                                                [[0],[1],[0]],
                                                [[0],[1],[1]],
                                                [[1],[0],[0]],
                                                [[1],[0],[1]],
                                                [[1],[1],[0]],
                                                [[1],[1],[1]]
                                            ]
                                        )
                        )
        self.assertTrue(np.array_equal(res8,                               
                                            [
                                                [[0],[0],[0]],
                                                [[0],[0],[1]],
                                                [[0],[1],[0]],
                                                [[0],[1],[1]],
                                                [[1],[0],[0]],
                                                [[1],[0],[1]],
                                                [[1],[1],[0]],
                                                [[1],[1],[1]]
                                            ]
                                        )
                        )
        self.assertTrue(np.array_equal(res9,
                                            [
                                                [[0],[0],[0],[0]],
                                                [[0],[0],[0],[1]],
                                                [[0],[0],[1],[0]],
                                                [[0],[0],[1],[1]],
                                                [[0],[1],[0],[0]],
                                                [[0],[1],[0],[1]],
                                                [[0],[1],[1],[0]],
                                                [[0],[1],[1],[1]],
                                                [[1],[0],[0],[0]],
                                                [[1],[0],[0],[1]],
                                                [[1],[0],[1],[0]],
                                                [[1],[0],[1],[1]],
                                                [[1],[1],[0],[0]],
                                                [[1],[1],[0],[1]],
                                                [[1],[1],[1],[0]],
                                                [[1],[1],[1],[1]]
   
                                            ]
                                        )
                        )
        print("test_cartesian_product CHECK")

    def test_cartesian_product_repeat(self):
        
        with self.assertRaises(Exception) as context:
            cartesian_product_repeat([0,1],-1)
        
        with self.assertRaises(Exception) as context: 
            cartesian_product_repeat([0,1],0)
        
        res2 = cartesian_product_repeat([0,1],1)
        res3 = cartesian_product_repeat([0,1],2)
        #res4 = cartesian_product_repeat([0,1],3)
        #res5 = cartesian_product_repeat([[0,2],[1,3]],2)
        
        self.assertTrue(np.array_equal(res2,[[[0]],[[1]]]))
        self.assertTrue(np.array_equal(res3,[[[0],[0]],
                                            [[0],[1]],
                                            [[1],[0]],
                                            [[1],[1]]]))
        '''
        self.assertTrue(np.array_equal(res4,[
                                                [[0],[0],[0]],
                                                [[0],[0],[1]],
                                                [[0],[1],[0]],
                                                [[0],[1],[1]],
                                                [[1],[0],[0]],
                                                [[1],[0],[1]],
                                                [[1],[1],[0]],
                                                [[1],[1],[1]]
                                            ]))
        
        self.assertTrue(np.array_equal(res5,[
                                                [[0],[2],[0],[2]],
                                                [[0],[2],[1],[3]],
                                                [[1],[3],[0],[2]],
                                                [[1],[3],[1],[3]],
                                            ]))
        '''
        print("test_cartesian_product_repeat CHECK")
    
    def test_xor_matrix(self):
        with self.assertRaises(Exception) as context: 
            xor_matrix([],[])
            
        with self.assertRaises(Exception) as context: 
            xor_matrix([[]],[[]])
            
        with self.assertRaises(Exception) as context: 
            xor_matrix([],[1])
        
        res0 = xor_matrix([0,1,1],[0,1,0])
        res1 = xor_matrix([[0,0,0],[1,1,1],[1,0,1]],[[0,0,0],[1,1,1],[1,0,0]])
        
        self.assertTrue(np.array_equal(res0,[0,0,1]))
        self.assertTrue(np.array_equal(res1,[[0,0,0],[0,0,0],[0,0,1]]))

        print("test_xor_matrix CHECK")
    
    def test_arr_to_tuple(self):
        res0 = arr_to_tuple([])
        res1 = arr_to_tuple([[]])
        res2 = arr_to_tuple([1,2,3]) 
        res3 = arr_to_tuple([1j,2j,1+3j])
        res4 = arr_to_tuple([[1,2,3],[1j,2j,1+3j]])
        
        self.assertEqual(res0,())
        self.assertEqual(res1,((),))
        self.assertEqual(res2,((1,0),(2,0),(3,0)))
        self.assertEqual(res3,
            (
                (0,1),(0,2),(1,3)
            )
        ) #(3,2)
        self.assertEqual(res4,
            (
                ((1, 0), (2, 0), (3, 0)), 
                
                ((0, 1), (0, 2), (1, 3))
            )
        ) #(2,3,2)
        
        print("test_arr_to_tuple CHECK")
    
    def test_tuple_to_arr(self):
        arr0 = []
        arr1 = [[]]
        arr2 = [1,2,3]
        arr3 = [1j,2j,1+3j]
        arr4 = [[1,2,3],[1j,2j,1+3j]]
        
        tmp0 = arr_to_tuple(arr0)
        tmp1 = arr_to_tuple(arr1)
        tmp2 = arr_to_tuple(arr2) 
        tmp3 = arr_to_tuple(arr3)
        tmp4 = arr_to_tuple(arr4)
        
        res0 = tuple_to_arr(tmp0)
        res1 = tuple_to_arr(tmp1)
        res2 = tuple_to_arr(tmp2)
        res3 = tuple_to_arr(tmp3)
        res4 = tuple_to_arr(tmp4)
        
        self.assertTrue(np.array_equal(res0,arr0))
        self.assertTrue(np.array_equal(res1,arr1))
        self.assertTrue(np.array_equal(res2,arr2))
        self.assertTrue(np.array_equal(res3,arr3))
        self.assertTrue(np.array_equal(res4,arr4))
        
        print("test_tuple_to_arr CHECK")
    
    def test_sum_matrix(self):
        #print(sum_matrix(1))
        #print(sum_matrix(2))
        #print(sum_matrix(3)) 
        return ()
    
    def test_set_to_list(self):
        tmp0 = sum_matrix(2)
        #print(set_to_list(tmp0))
        
        

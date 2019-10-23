# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:28:46 2019

@author: jubin
"""

# x = list(map(int, str(x))) 

"""
x= 568
x = list(map(int, str(x))) 
x.insert(0,2)
x.pop(x[1])
y = int("".join(map(str,x)))
"""

def max_num(num):
    
    max_val = num
    res = [int(num) for num in str(num)]
    k = 5
    
    for i in range(len(res)+1):
        res.insert(i,k)
        val = int("".join(map(str,res)))
        print('val',i,': ',val, sep= " ")
        if val > max_val:
            max_val = val
        res.remove(res[i])
        
    return max_val
        

num = 670
max_num(num)     
        
        
        
    
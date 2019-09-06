# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:48:06 2019

@author: jubin
"""

def search(x, y):
    
    d1 ={}
    
    d2 ={}
    
    l1 = []
    for i in x:
        
        if i not in d1:
            d1[i] = 1
            
        else:
            d1[i] += 1
            
    for j in y:
        
        if j not in d2:
            d2[j] = 1
            
        else:
            d2[j] += 1
            
            x
    new_dict={}
    for key in d1:
        if key in d2:
            new_dict[key] = abs(d1[key] - d2[key])
    
    for key in d1.keys(): 
        if not key in d2: 
  
        # Printing difference in 
        # keys in two dictionary 
            l1.append(d1[key])
    for key in d2.keys(): 
        if not key in d1: 
  
        # Printing difference in 
        # keys in two dictionary 
            l1.append(d2[key])
    for val in l1:
        if val >3:
            return 'NO'
    
    
    # new_dict = {k: abs(d1[k] - d2[k]) for k in d1.keys() & d2.keys()}
    
    for key, value in new_dict.items():
        
        if value >3:
            return 'NO'
    
    return 'YES'

search('aaaabbbbccceeeeee','aabeee')
    
    
    
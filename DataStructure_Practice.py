# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:11:55 2019

@author: jubin
"""

"""
# Most frequently occuring item
x = [1,3,1,3,2,1,3,2,2,2,2,2,2,1]
max_item = None

def most_freq(input):
    count_item = -1
    max_item = None
    count = {}
    
    
    for i in input:
        if i not in count:
            count[i] = 1
        else:
            count[i]+=1
        if count[i] > count_item:
            count_item = count[i]
            max_item =i
    return max_item



most_freq(x)   
"""

# a= [1,2,3]
# len(a)

"""
# Common elements in two sorted arrays

def cmn_elements(A,B):
    p1 = 0
    p2 = 0
    result = []
    while p1 < len(A) and p2 < len(B):
        
        if A[p1]== B[p2]:
            result.append(A[p1])
            p1 += 1
            p2 += 1
            
        elif A[p1] > B[p2]:
            p2 += 1
        
        else:
            p1 += 1
    
    return result

x= [1,2,3,4,9]
y = [2,5,6,7,9]

cmn_elements(x, y)
        
"""

"""
# One array is a rotation of another

def rot_array(A,B):
    
    if len(A) != len(B):
        return False
    
    key = A[0]
    
    key_pos = -1
    
    for i in range( 0, len(B)):
        if B[i] == key:
            key_pos = i
            break
    if key_pos == -1:
        return False
    
    for i in range(0, len(A)):
        j = (key_pos + i) % len(A)
        if A[i] != B[j]:
            return False
    return True

A = [1,2,3,4,5,6,7]
B = [4,5,6,7,1,2,3]

rot_array(A,B)

"""

# reversing a array 

"""
class Str_Conv:
    def __init__(self, input):
        self.input = input
    def s_con(self):
        x  = list(self.input)
        x = x[::-1] 
        str ="".join(x)    
        return str
    
p = Str_Conv("apple")

p1 = Str_Conv("radar")

p2 = Str_Conv("Local")

p2.s_con()

"""

"""
def reverse(str):
    rev =""
    
    for var in str:
        rev = var +str
    return rev
    

reverse("apple")

def revers(strs):
    x = []
    count =1
    for i in range(0,len(strs)):
        x.append(strs[len(strs)-count])
        count += 1
    x="".join(x)
    return x
    
    
    
revers("apple")
"""

# Anagram
'client eastwood' 'old west action'

"""

def anagrm(s1, s2):
    
    s1 = s1.replace(' ','').lower()
    s2 = s2. replace(' ','').lower()
    
    # return sorted(s1)==sorted(s2)
    if len(s1) != len(s2):
        return False
         
    count = {}
    
    for i in s1:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
            
    for j in s2:
        if j not in count:
            count[j] = 1
        else:
            count[j] -= 1
    
    for k in count:
        if count[k] != 0:
            return False
    
    return True

anagrm('clint eastwood', 'old west action')
            
count = {'c':2}
  
"""

"""
# Array Pair sum

def pair_sum(ar, key):
    
    if len(ar)< 2:
        return 
    
    seen = set()
    output = set()
    
    for i in ar:
        tar = key-i
        if tar not in seen:
            seen.add(i)
        else:
            output.add(((min(i,tar)),max(i,tar)))
    
    print('\n'.join(map(str,list(output))))
    
pair_sum([1,3,2,2],4)

"""


"""

# Palo alto Program"

import re


p =list("jubin12345")
l1 = p[0:5]
l2 = p[5:10]
#l2 = l2[::-1]
nw_ar = []
count =1    
for i in range(0, len(l2)):
    nw_ar.append(l2[len(l2)-count])
    count += 1

x = l1 + nw_ar

x = "".join(x)


"""

# Find the missing element
# 1st Menthod
"""
def finder(ar1, ar2):


    ar1.sort()
    ar2.sort()

    for num1, num2 in zip(ar1, ar2):
        if num1 != num2:
            return num1
        
    return ar1[-1]
"""

"""
# 2nd Method

import collections
def finder(ar1,ar2):
    d = collections.defaultdict(int)
    
    for num in ar2:
        d[num] += 1

    for num in ar1:
        if d[num] == 0:
            return num
        
        else:
            d[num] -= 1
    
 
arr1 = [1,2,3,4,5,6,7]
arr2 = [3,7,2,1,4,6]

finder(arr1, arr2)

"""


"""
# Largest continuous sum

def lar(arr):
    
    if len(arr)==0:
        return 0
    max_sum = current_sum = arr[0]
    for num in arr[1:]: 
        current_sum = max(current_sum, num)
        max_sum = max(current_sum, max_sum)
    
    return max_sum

lar([1,2,-1,3,4,10,10,-10,-1])    
"""

# word reversing
"""
-- assign a pointer and then iterate it until finding a space
-- then again change the pointer, to a new position

"""

def rev_word(s):
    
    word = []
    space=[' ']
    length = len(s)
    
    i =0
    
    while( i< length):
        
        if s[i] not in space:
            
            w_start =i
            
            while(i< length and s[i] not in space):
                
                i += 1
            
            word.append(s[w_start:i])
        
        i += 1
        
    return " ".join(reversed(word))

rev_word(' Hello John  How are you ')







    






        
                 
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:36:02 2019

@author: jubin
"""

# sorting a list of tuples

list1 = [('eggs', 5.25),('honey', 9.70),('carrots', 1.10),('peaches', 2.45)]

print(list1.sort(key = lambda x:x[0]))


# Sortuing a list of dictionaries using lambda

import pprint as pp

list2 = [{'make': 'Ford', 'model': 'Focus', 'year' : 2013},
         {'make': 'Tesla', 'model': 'X', 'year' : 1999},{'make': 'Mercedes', 'model': 'T', 'year' : 2008}]


l = sorted(list2, key = lambda x: x['year'], reverse = True)

pp.pprint(l)


# Filter function and using in junction with lambda operation

def fun(variable):
    letters = ['a', 'e', 'i', 'o', 'u']
    
    if variable in letters:
        return True
    else:
        return False

# sequence 
sequence = ['p','q','l', 'n']

# using filter function 
filtered = filter(fun, sequence)

print('The filtered letters are:') 
for s in filtered: 
    print(s)
    
# with lambda operator
seq = [0,1,2,3,5,8,13]

result = filter(lambda x:x % 2, seq)    

print(list(result))

# result contains even numbers of the list

result = filter(lambda x:x % 2==0, seq)
print(list(result))

# Output of Python Programs
a = True
b = False
c = False
  
if a or b and c: 
    print ("GEEKSFORGEEKS")
else: 
    print ("geeksforgeeks")



# Explanation of map function
    
n = [4,3,2,1]

def square(li):
    
    li1 = []
    
    for i in li:
        li1.append(i**2)
    
    return li

x = list(map(square,n))


x = list(map(lambda x: x**2, n))
    
li = [i*2 for i in range(0,10)]
    







    
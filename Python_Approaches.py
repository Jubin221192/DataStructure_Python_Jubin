# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 00:18:40 2019

@author: jubin
"""

# Copy operator in List
# In this scenario the entire list object reference is copied.

list1 = [[1,2,3],[4,5,6],[7,8,9]]

list2 = list1

print(list1)

print(list2)

print("id of list1 -> ", id(list1))
print("id of list2 -> ", id(list2))

"""
id of list1 ->  1891489018696
id of list2 ->  1891489018696
"""

list1.append([10,11,12])

list2[2][1] = 'AA'

# SHALLOW COPY

"""
Shallow copy does not create copy of nested object instead it stores the 
reference  of the original elements(or nested objects)
"""


import copy
old_list = [[1,2,3],[4,5,6],[7,8,9]]
new_list = copy.copy(old_list)

print("old_list -> ", id(old_list))
print("new_list -> ", id(new_list))

"""
old_list ->  1891492224904
new_list ->  1891492234760

"""

# Adding new nested object using shallow copy

new_list[2][2]= 'dd'

print(old_list)
print(new_list)

new_list.append(['kk',12,25])

"""

print(old_list)
print(new_list)
[[1, 2, 3], [4, 5, 6], [7, 8, 'dd']]
[[1, 2, 3], [4, 5, 6], [7, 8, 'dd'], ['kk', 12, 25]]


"""

# Deep Copy

"""

Deep copy creates a brand new object and subsequently creates a copy of nested
objects

"""

import copy
old_f = [[1,1,1],[2,2,2],[3,3,3]]
new_f = copy.deepcopy(old_f)

print("old list: ", old_f)
print("new list: ", new_f)

print("id of old list ->", id(old_f))
print("id of first list ->", id(new_f))

# Adding a new list
old_f[1][0] = 'BB'

# Functional Programming

def calc(f,x,y):
    return f(x,y)

def add(x,y):
    return x+y

def sub(x,y):
    return x-y

calc(add,10,20)
    
# Lambdas
# Annonymous functions

def calc(f,x,y):
    return f(x,y)

calc(lambda x,y: x+y, 10, 20)
calc(lambda x,y: x-y, 20, 10)

#Function way to increment
def incr(x):
    return x + 1

def incr_each(elements):
    result = []
    for ele in elements:
        result.append(incr(ele))
    
    return result


incr_each([1,2,3,4])

print(map(incr, [5,6,7]))
x = map(lambda x:x+1,[5,6,7])
x


# Less contrived example

results = []
elements = [1,2,3,4]

for ele in elements:
    results.append(len(ele))
    
# Coding challenge 1
    
price = 5000
price -= 0.1*price

price -= 0.05 * price


def stu_dis(price):
    
    price -= (0.1*price)
    
    print("Student price", price)
    
    return (price)

def reg_buy(price):
    
    price = stu_dis(price)
    price -= (0.05 * price)
    
    print("REGULAR price", price)    
    return (price)

def price_cal(f,price):
    return f(price)    


price_cal(reg_buy,5000)


# String Formatting

numbers = [4,5,6]

newstring = "Numbers:{0},{1},{2}".format(numbers[0],  numbers[1], numbers[2])

print(newstring)


newstring = "This is a string"
print(newstring.endswith("string"))


# Few coding chalenges

products ={'chair': 40, 'Table' : 50, 'Sofa' : 200, 'Bed' : 150}

prod_nm = input('Enter a product :')

print(products.get('Bed'))

if prod_nm in products:
    print(products.get(prod_nm))
    
else:
    print('Product not found')
    

# list out all the odd numbers in 
    

ll = [x for x in range(1, 101) if x % 2 != 0]
print(''.join(str(ll)))


a= 9
b= 6
c= "a is greater"

res = c if a >b else 'none'

print(res)

my_dict = {'a': 'jill', 'b': 'tom', 'c': 'tim'}
for key,value in my_dict.items():
    print(key + ', ' + value)

x = dict((value,key) for key,value in my_dict.items())



























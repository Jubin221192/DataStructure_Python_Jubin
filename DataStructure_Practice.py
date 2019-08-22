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


# word reversing
"""
-- assign a pointer and then iterate it until finding a space
-- then again change the pointer, to a new position


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

"""


# String compression
"""
Here I have simple compare characters at two subsequent positions

'AAAABBBDDDDDD'

def comp(wor):
    
    r = ""
    if len(wor) == 0:
        return ""
    
    if len(wor) == 1:
        return wor+"1"
    
    cnt =1
    i = 1
    # last = s[0]
    
    while i < len(wor):
        
        if wor[i]==wor[i-1]:
            cnt +=1
        
        else:
             r = r + wor[i-1] + str(cnt)
             cnt = 1
        
        i += 1
    
    r = r + wor[i-1] + str(cnt)
    return r
    
comp("AAABBCCCC")
"""        


"""
import sys

def sim(word): 
    
    if len(word) == 0:
        sys.exit("Null string")
    
    if len(word) == 1:
        return False
    
    i = 1
    
    while i < len(word):
        if word[i]==word[i-1]:
            return False
        else:
            i += 1
    return True


sim("ABBD")
"""

"""
import pandas as pd

pd.read_csv('C:/Users/jubin/Downloads/train.csv')


def to_string(given_array):
    list_rows = []
    for row in given_array:
        list_rows.append(str(row))
    return '[' + ',\n '.join(list_rows) + ']'

li = [2,1,0,1]

to_string(li)
"""
"""
# Mine Sweeper

def mine(bomb, rows, clm):
    
    field = [[ 0 for i in range(clm)] for i in range(rows)]
    
    for bm in bomb:
        (r_i, c_i)  =  bm
        field[r_i][c_i] = -1
        for i in range(r_i-1, r_i+2):
            for j in range(c_i-1, c_i+2):
                if(0 <= i < rows and 0 <= j < clm and field[i][j] != -1):
                    field[i][j] += 1
    
    return field


mine([[0, 2], [2, 0]], 3, 3) 

def to_string(array):
    list_rows = []
    for row in x:
        list_rows.append(str(row))
    return '[' + ',\n '.join(list_rows) + ']'   

to_string(x)                 
"""

    # Palindrome

"""
def chPal(string):
    ch = list(string)
    l = len(ch)-1
    i=0
    pa = []
    while(i < len(ch)):
        pa.append(string[l-i])
        i+=1
    
    f_pa = "".join(pa)
    
    return f_pa == string

chPal('snehal')
"""

"""
# Reverse integer
    
def rev_int(num):
    ans = 0
    last_digit = 0
    negativeFlag = False
    if (num < 0): 
        negativeFlag = True
        num = -num  
    while num != 0:
        last_digit = num % 10
        ans = ans*10 + last_digit
        num //= 10
    
    return -ans if(negativeFlag == True) else ans
    

rev_int(258)
        
"""

# Reversing a dictionary

"""
list=[2,10,8]
list.pop()
thistuple = {"apple" : 2, "banana": 10, "cherry":8}

for c in list:
    print(c)
    if c in thistuple:
        print("true")



rev_dict = {y:x for x,y in thistuple.items()}     
pp = dict([(v, k) for k, v in thistuple.items()])
"""

# Balanced Parenthesis check

"""
Stack -> FILO [ first in first out]

SOLUTION

This is a very common interview question and is one of the main ways to check your knowledge of using Stacks!
We will start our solution logic as such:

First we will scan the string from left to right, and every time we see an opening parenthesis we push it to a stack, 
because we want the last opening parenthesis to be closed first. (Remember the FILO structure of a stack!)

Then, when we see a closing parenthesis we check whether the last opened one is the corresponding closing match, 
by popping an element from the stack. If it’s a valid match, then we proceed forward, if not return false.

Or if the stack is empty we also return false, because there’s no opening parenthesis associated with this closing one. 
In the end, we also check whether the stack is empty. If so, we return true, 
otherwise return false because there were some opened parenthesis that were not closed.

"""

"""
# First Approach

def balance_check(s):
    chars = []
    matches = {')':'(',']':'[','}':'{'}
    for c in s:
        if c in matches:
            if chars.pop() != matches[c]:
                return False
        else:
            chars.append(c)
    return chars == []        
    pass


balance_check('[](){([[[]]])}')
"""
                
# Second Approach

"""
def bal_check(s):
    
    #chceking even number of elements
    if(len(s)%2)!=0:
        return False
    
    stack = []
    
    # opening parethesis
    opening = set ('({[')
    
    matching_pairs = set([('(',')'),('{','}'),('[',']')])
    
    for val in s:
        
        if val in opening:
            stack.append(val)
        
        else:
            
            # check the parenthesis in the stack or not
            if len(stack) == 0:
                return False
            
            #last opening parenthesis:
            
            last_open = stack.pop()
            
            #checking whether it has a closing mtch or not
            
            if (last_open,val) not in matching_pairs:
                return False
    
    return len(stack) == 0

bal_check('[[')
    
"""

# Inmplementing a Queue

"""


Check if Queue is Empty
Enqueue
Dequeue
Return the size of the Queue


"""
"""
class Queue(object):
    
    def __init__(self):
        self.item = []
    
    def isempty(self):
        return self.item == []
    
    def enqueue(self,e):
        self.item.insert(0,e)
        
    def dequeue(self):
        return self.item.pop()
    
    def size(self):
        return len(self.item)
    
    pass


q = Queue()
q.isempty()

q.enqueue(9)
q.size()
q.dequeue()


list([1,2,3])


# Implementing queue using two stacks:

class Que2stack(object):
    
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def enqueue(self,e):
        x = self.stack1.append(e)
        return x
        
    def dequeue(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
     
    pass
        

r = Que2stack()

r.enqueue(9)

r.dequeue()

"""

"""
#     
 
ar1 = 'good is sam' 

x= ar1.split(' ')

y =[]
count = 1

for i in range (0,len(x)):
    
    y.append(x[len(x)-count])
    
    count += 1
    
y = " ".join(y)

"""

"""
-----------------
Sequential search
-----------------

"""
"""

def ord_seq_search(arr, ele):
    
    pos = 0
    
    found = False
    stopped = False
#   while pos < len(arr) and not found and not stopped:
    while pos < len(arr):
    
        if arr[pos] == ele:
            found = True
            return found
        
        else:
            if arr[pos] > ele:
                return stopped
                
            else:
                pos += 1
            
    return found



ord_seq_search([5,1,2,9,8,7],8)

ord_seq_search([1,2,5,7,8,9,12,14],6)
    
        
"""
#-----------------
#Binary search
#-----------------

"""


def binary_search(arr,ele):
    
    found = False
    first = 0
    last = len(arr) - 1
    
    
    while first <= last and not found:
        mid = (first + last)//2
        
        if arr[mid] == ele:
            found = True
            
        else:
            if ele < arr[mid]:
                last = mid -1
            
            else:
                first = mid + 1
    
    return found


# How to run it recursively
       
def recur_binary_search(arr,ele):
    
    first = 0
    last = len(arr) - 1
    
    while first <= last:
        mid = (first + last)//2 
        
        if arr[mid] == ele:
            return True
            
        else:
            if ele < arr[mid]:
                return recur_binary_search(arr[:mid], ele)
            
            else:
                return recur_binary_search(arr[mid+1:], ele)
    
    return False 
            
        
recur_binary_search([1,2,3,4,5,6,7,8,9,10],11)
            

        

d = {'x':1,'y':5,'z':6}


m = dict(zip(d.values(),d.keys()))
print(m)
        
 
        
# First letter capitaization

def letter_capitalization(string):
    
    cap_str = []
    for val in string.split(' '):
        cap_str.append(val[0][0].upper() + val[1:])
    
    return ' '.join(cap_str)


string = 'one day i will become what I dreamt of'

letter_capitalization(string)


"""
# checking vowels

"""
def check_vowels(string):
    count = 0 

    string = string.lower()

    for char in string:

        if char in 'aeiou':

            count += 1

    return count


check_vowels("Galleo gallili")


# Iterative methodology to print a pyramid

int(((2*4)-1)/2)

def pyramid(n):
     mid_p = int(((2*n)-1)/2)
     #li = []
     for row in range(0,n):
         # l ='
         li = []
         for col in range(0, ((2*n)-1)):
             if (mid_p - row) <= col and (mid_p + row) >= col:
                 li.append('#')
             
             else:
                 li.append(' ')
         print(''.join(li))
         
         li=li.clear()
    

pyramid(3)
"""

"""             
             
# Pyramid problem using recursion

def pyramid(n, row =0, lev = []):
    
    if row==n:
        return
    
    if len(lev) == ((2*n)-1):
        print(''.join(lev))

        lev = lev.clear()
        return pyramid(n, row +1)
    
    mid_point = int(((2*n)-1)/2)
    
    if (mid_point - row) <= len(lev) and (mid_point + row) >= len(lev):
        lev.append('#')
    
    else:
        lev.append(' ')
    
    return pyramid(n,row,lev)

pyramid(9)

#Logic for printing a triangle
def pattern1(n):
    
    
    for i in range(0,n):
        for j in range(0,n):
            
            if(j<=i):
                print('#',end=' ')
            else:
                print(' ', end =' ')
        print('\n')
        


#Printing a triangle using recurssion
def triangle(n, row=0, lev = []):
    
    if row == n:
        return
    
    if len(lev) == n:
        print(''.join(lev))
        
        lev= lev.clear()
        return triangle(n, row+1)
    
    if len(lev) <= row:
        lev.append('#')
    
    else:
        lev.append(' ')
    
    return triangle(n,row,lev)
    
    
triangle(5)
"""

# Implement your function below
"""
    
import copy
def to_string(given_array):
    list_rows = []
    for row in given_array:
        list_rows.append(str(row))
    return '[' + ',\n '.join(list_rows) + ']'
    
"""     


"""
Output

[[1, 2, 3], 
 [4, 5, 6], 
 [7, 8, 9]]

===

    [[7, 4, 1], 
     [8, 5, 2], 
     [9, 6, 3]]


"""
"""


to_string(a1)        

# Array roation to 90
# The assumption is that the number of columns is equal to number of rows = n 

def rotate_sub(i,j,n):
    return j,n-1-i


def rotate(given_arr, n):
    
    rot = copy.deepcopy(given_arr)
    
    for i in range(n):
        for j in range(n):
            
            (new_i, new_j) =rotate_sub(i,j,n)
            
            rot[new_i][new_j] = given_arr[i][j]
            
    return rot

rotate(a1,3)
"""

"""

Spiral Matrix


"""
"""
class Solution:
    # @param matrix, a list of lists of integers
    # @return a list of integers
    def spiralOrder(self, matrix):
        if matrix == []: return []
        up = 0; left = 0
        down = len(matrix)-1
        right = len(matrix[0])-1
        direct = 0  # 0: go right   1: go down  2: go left  3: go up
        res = []
        while True:
            if direct == 0:
                for i in range(left, right+1):
                    res.append(matrix[up][i])
                up += 1
            if direct == 1:
                for i in range(up, down+1):
                    res.append(matrix[i][right])
                right -= 1
            if direct == 2:
                for i in range(right, left-1, -1):
                    res.append(matrix[down][i])
                down -= 1
            if direct == 3:
                for i in range(down, up-1, -1):
                    res.append(matrix[i][left])
                left += 1
            if up > down or left > right: return res
            direct = (direct+1) % 4
            
a1 = [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]]

obj = Solution()
obj.spiralOrder(a1)
    

10//2

# Merchant sock problem

import collections
def sock_mer(ar):
    d = collections.defaultdict(int)
    
    for k in ar:
        d[k] += 1
        
    cnt = 0
    for ele in d.values():
        cnt += ele//2
    
    return cnt

sock_mer([10,20,20,10,30])

x=2
pa = set()

"""

"""

def socks_par(ar):
    
    pa = set()
    count = 0
    for k in ar:
        if k in pa:
            pa.remove(k)
            count +=1
        else:
            pa.add(k)
    
    return count


socks_par([10,20,20,10,30])
    
    
            
            
# Jumping on the clouds

def jump_on_clouds(ar):
    
    
    jump = 0
    i =0
    while i < len(ar)-1:
        
        if (i+2 == len(ar)) or (ar[i+2]==1):
            i += 1
            jump +=1
        
        else:
            i += 2
            jump += 1

    return jump



jump_on_clouds([0,1,0,0,1,0])        


# Repeated Strings

def repeat_strings(s, n):
    
    count = 0
    value=0
    ar = []
    i =0
    s = list(s)
    while i < n:
        
        ar.append(s[count])
        if ar[i]=='a':
            value += 1
            
        count += 1
        if (count == len(s)):
            count = 0
        
        i += 1
    
    
    
    return  value
            

repeat_strings('abaaa',10)

# Complete the repeatedString function below.
def repeatedString(s, n):
    n_of_a = 0
    for i in s:
        if i == 'a':
            n_of_a += 1
    res = int(n_of_a * (n / len(s)))
    for i in s[:n % len(s)]:
        if i == 'a':
            res += 1
    return res
            
            
"""        
"""          
def rans_note(mag, note):
    
    mag = mag.split(' ')
    note = note.split(' ')
    
    d = {}
    
    for i in mag:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    
    for j in note:
        if d.get(j) is None or d[j] == 0:
            return 'No'
        else:
            d[j] -= 1
    
    return 'Yes'
            

rans_note('ive got a lovely bunch of coconuts', 
          'ive got some coconuts')
"""
"""
def twoStrings(s1, s2):

    d= {}
    ar =[]

    for i in s1:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    for j in s2:
        if d.get(j) is not None and d[j] > 0:
            ar.append(j)
    if (len(ar) == 0):
        return 'No'
    
    return 'YES'
           
            

s1 = list('Hello')
s2 = list('World')
twoStrings(s1, s2)

"""
    
# Anagrams

"""  
def anagram(s):
    n = len(s)
    res = 0
    for l in range(1, n):
        cnt = {}
        for i in range(n - l + 1):
            subs = list(s[i:i + l])
            subs.sort()
            subs = ''.join(subs)
            if subs in cnt:
                cnt[subs] += 1
            else:
                cnt[subs] = 1
            res += cnt[subs] - 1
    return (res)

ar = 'abba'
anagram(ar)
"""

"""

# Counting triplets

def countTriplets(arr, r):
        # initialize the dictionaries
        r2 = {}
        r3 = {}
        count = 0

        # loop throgh the arr itens
        for k in arr:
                # if k in r3 indicates the triplet already completed,
                # the count need be incremented
                if k in r3:
                        count += r3[k]

                # if k in r2, it is the secound number of the triplet,
                # your successor (third element k*r) need be added or incremented in the r3 dict
                # because is a potencial triplet 
                if k in r2:
                        if k*r in r3:
                                r3[k*r] += r2[k]
                        else:
                                r3[k*r] = r2[k]

                # else, k is the first element of the triplet, so,
                # your seccessor (secound element k*r) need be added or incremented in the r2 dict
                # because is a potencial triplet
                if k*r in r2:
                        r2[k*r] += 1
                else:
                        r2[k*r] = 1

        return count

import collections
def countTriplets(arr, r):
    # stores number of tuples with two elements that can be formed if we find the key
    potential_two_tuples = collections.defaultdict(int) 
    # stores number of tuples with three elements that can be formed if we find the key
    potential_three_tuples = collections.defaultdict(int)
    count = 0
    for k in arr:
        # k completes the three tuples given we have already found k/(r^2) and k/r  
        count += potential_three_tuples[k]
        # For any element of array we can form three element tuples if we find k*r given k / r is already found. Also k forms the second element.
        potential_three_tuples[k*r] += potential_two_tuples[k]
        # For any element of array we can form two element tuples if we find k*r. Also k forms the first element.
        potential_two_tuples[k*r] += 1 
    return count

countTriplets([1,2,2,4] , 2)
    
"""

# find the pairs with given sum in an array:

"""
def pairs(arr, gm):
    
    arr = sorted(arr)
    l = len(arr) -1
    p = 0
    arr_sum = []
    
    
    
    while p< l:
        
        if (arr[p] + arr[l]) > gm:
            
            l -= 1
            
        elif (arr[p] + arr[l]) < gm:
            p +=1
            
        else:
            arr_sum.append([arr[p],arr[l]])
            p += 1
        
    
    return(len(arr_sum))
           

pairs([5,8,3,4,2,6,10,7,1,9], 11)
"""           


# Making Anagrams
"""
def remAnagram(str1, str2): 
  
    # make hash array for both string  
    # and calculate 
    # frequency of each character 
    CHARS = 26
    count1 = [0]*CHARS 
    count2 = [0]*CHARS 
  
    # count frequency of each character  
    # in first string 
    i = 0
    while i < len(str1): 
        count1[ord(str1[i])-ord('a')] += 1
        i += 1
  
    # count frequency of each character  
    # in second string 
    i =0
    while i < len(str2): 
        count2[ord(str2[i])-ord('a')] += 1
        i += 1
  
    # traverse count arrays to find  
    # number of characters 
    # to be removed 
    result = 0
    for i in range(26): 
        result += abs(count1[i] - count2[i]) 
    return result 
  
# Driver program to run the case 

"""
"""    
if __name__ == "__main__": 
    str1 = "bcadeh"
    str2 = "hea"
    print(remAnagram(str1, str2))
"""

#remAnagram("bcadeh", "hea")


#mak_anagrams('cde', 'dcf')

# Alternating Characters
"""
def alt_char(s):
    s = s.lower()
    d_cha = 0
    for i in range(0, len(s)-1):
        if s[i] == s[i+1]:
            d_cha +=1
    return d_cha

alt_char('AAABBB')
"""    


# Sherlock and the Valid String

"""    
from collections import Counter
 
 
def isValid(S):
    char_map = Counter(S)
    char_occurence_map = Counter(char_map.values())
    print('char_map',char_map,'\n')
    print('char_occurence_map',char_occurence_map,'\n')    
    if len(char_occurence_map) == 1:
        return True
 
    if len(char_occurence_map) == 2:
        for v in char_occurence_map.values():
            if v == 1:
                return True
 
    return False
 
   
S= 'aabbcd'
isValid(S)
"""
"""
if isValid(S):
    print ("YES")
else:
    print ("NO")
"""

"""
from collections import Counter

def isValid(S):
    c = Counter(S)
    d = Counter(c.values())
    print('c: ',c,'\n')
    print('Counte(c.values): ',d,'\n')
    distinct_counts = list(set(d.elements()))
    print('distinct_counts: ', distinct_counts)
    if len(d) == 1:
        return "YES"
    elif len(d) == 2:
        the_greater_count = max(distinct_counts[0], distinct_counts[1])
        the_lesser_count = min(distinct_counts[0], distinct_counts[1])
        if the_greater_count - the_lesser_count == 1:
            if d[the_greater_count] == 1:
                return "YES"
            elif d[the_lesser_count] == 1:
                return "YES"
            else:
                return "NO"
        elif d[the_lesser_count] == 1:
            return "YES"
        else:
            return "NO"
    else:
        return "NO"
        
isValid('abccc')
"""

# Special String Again , substring Pallindrome


def substrCount(n, s):
    l = []
    count = 0
    cur = None

# 1st pass
    for i in range(n):
        if s[i] == cur:
            count += 1
        else:
            if cur is not None:
                l.append((cur, count))
            cur = s[i]
            count = 1
    l.append((cur, count))

    ans = 0
		
# 2nd pass
    for i in l:
        ans += (i[1] * (i[1] + 1)) // 2

# 3rd pass
    for i in range(1, len(l) - 1):
        if l[i - 1][0] == l[i + 1][0] and l[i][1] == 1:
            ans += min(l[i - 1][1], l[i + 1][1])

    return ans

substrCount(7,'abcbaba')

# Minimum swaps to sort the array using graph concept

def minSwaps(arr):
    
    counter = 0
    length = len(arr)
    arr_dict = {}
    
    for i, item in enumerate(arr):
        
        arr_dict[item] = i
        
    checked = [False]*length
    
    for key, value in sorted(arr_dict.items(), key = lambda x: x[0]):
        
        if checked[key] or key==value:
            continue
        
        c_count = 0
        value = key
        
        while not checked[value]:
            
            checked[value] = True
            value = arr_dict[value]
            c_count += 1
            
        counter += c_count -1
        
    return counter

print(minSwaps([0,2,3,4,1,6,5]))
        
# sum of consecutive numbers:

def solution(N):
    
    count = 0
    n= 2
    
    while (2*N - (n**2) + n) > 0:
        a = (2*N - n**2 + n)/(2*n)
        
        if a -int(a)==0:
            print (a,n)
            count += 1
            
        n += 1
        
    return count

solution(10)


        
# Tim Sort implementation : which is a combination of insertion and merge sort

# Insertion sort
def insertion_sort(A):
    for i in range(1, len(A)):
        curNum = A[i]
        
        for j in range(i-1,0 -1):
            
            if A[j] > curNum:
                A[j+1] = A[j]
            else:
                A[j+1]= curNum
                
    
# Merge sort
                
                
def merge_sort(aArr, bArr):
    
    a= 0
    b=0
    cArr = []
    
    while a< len(aArr) and b< len(bArr):
        if aArr[a]< bArr[b]:
            cArr.append(aArr[a])
            a += 1
            
        elif aArr[a] > bArr[b]:
            cArr.append(bArr[b])
            b +=1
        
        else:
            cArr.append(aArr[a])
            cArr.append(bArr[b])
            
            a += 1
            b += 1
            
    while a<len(aArr):
        cArr.append(aArr[a])
        a += 1
    while b<len(bArr):
        cArr.append(bArr[b])
        b +=1
        
    return cArr


# Tim sort


def timsort():
    
    for  x in range (0,len(arr), Run):
        
        arr[x:x+Run] = insertion_sort([x:x+Run])
        
    
    Runinc = Run
    
    while Runinc < len(arr):
        
        for x in range(0,len(arr), 2*Runinc):
            
            arr[x: x + 2 * Runinc] = merg_sort(arr[x: x+ Runinc], arr[x+Runinc: x + 2*Runinc])
            
        Runinc = Runinc *2
    
        
    
            
        
        


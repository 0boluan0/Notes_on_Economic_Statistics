#################
## EXAMPLE: change value in a list and appending a value to a list
###############
# L = [2, 4, 3]
# print(L)
# L[1] = 5
# print(L)
# L = L.append(5)
# print(L)


############# YOU TRY IT #####################
# What is the value of L1, L2, L3, and L after these commands?
# L1 = ['re']
# L2 = ['mi']
# L3 = ['do']
# L4 = L1 + L2
# L3.append(L4)
# L = L2.append(L3)

#################################################


############### YOU TRY IT #######################
# Write a function that meets the specification:
def make_ordered_list(n):
    """ n is a positive int
    Returns a list containing all ints in order 
    from 0 to n (inclusive)
    """
    # your code here

    
#print(make_ordered_list(6))  # prints [0, 1, 2, 3, 4, 5, 6]

#####################################################


############ YOU TRY IT ###############
def remove_elem(L, e):
    """ 
    L is a list
    Returns a new list with elements in the same order as L
    but without any elements equal to e. 
    """
    # your code here

  
# L = [1,2,2,2]
# print(remove_elem(L, 2))    # prints [1]
# L = [1,2,2,2]
# print(remove_elem(L, 1))    # prints [2,2,2]
# L = [1,2,2,2]
# print(remove_elem(L, 0))    # prints [1,2,2,2]


#######################################

#################
## EXAMPLE: string-list ops
#################
# s = "I<3 cs and u?"		
# L = list(s) 	
# L1 = s.split(' ')	
# L2 = s.split('<')
# print(L)
# print(L1)
# print(L2)

# L = ['a','b','c']
# A = ''.join(L)	
# print(A)
# B = '_'.join(L)	
# print(B)

# Lnum = [1, 2, 3]
# # n = ''.join(Lnum)  # this line throws an error
# Lnum = ['1', '2', '3']
# n = ''.join(Lnum)
# print(n)

####### YOU TRY IT ###################
# Write a function that meets this specification
def count_words(sen):
    """ sen is a string representing a sentence 
    Returns how many words are in sen (i.e. a word is a 
    a sequence of characters between spaces. """
    # your code here


# s = "Hello it's me"
# print(count_words(s))   # prints 3

# s = "I just took a DNA test turns out I'm 100% splitting strings"
# print(count_words(s))   # prints 12

###########################################


###################
## EXAMPLE: sorting/reversing ops
##################
# L=[9,6,0,3]
# a = sorted(L)
# print(a)
# print(L)
# a = L.sort()
# print(a)
# print(L)
# L.reverse()
# print(L)


############## YOU TRY IT #################
# Write a function that meets this specification
def sort_words(sen):
    """ sen is a string representing a sentence 
    Returns a list containing all the words in sen but
    sorted in alphabetical order. """
    # your code here


# s = "look at this photograph"
# print(sort_words(s))    # prints ['at', 'look', 'photograph', 'this']

# s = "now this is a story all about how my life got flipped turned upside down"
# print(sort_words(s))

##########################################


##############
## Loops over lists
################
def square_list(L):
    for i in range(len(L)): 
        L[i] = L[i]**2

# print(square_list([2,3,4]))  # prints None

# Lin = [2,3,4]
# print("before fcn call:",Lin)
# square_list(Lin)
# print("after fcn call:",Lin)   # mutated L



##############
## TRICKY EXAMPLE 1: append to L white iterating over range(L)
##############
# L = [1,2,3,4]
# for i in range(len(L)):
#     L.append(i) 
#     print(L)


##############
## TRICKY EXAMPLE 2: append to L while iterating over L
## this leads to an infinite loop
##############
# L = [1,2,3,4]
# i = 0
# for e in L:
#     L.append(i)
#     i += 1
#     print(L)

## extend a list
# L1 = [2,1,3]
# L2 = [4,5,6]
# L3 = L1 + L2
# L1.extend([0,6])
# L2.extend([[0,2],[7,9]])

##############
## TRICKY EXAMPLE 3: combining
##############
# L = [1,2,3,4]
# for e in L:
#     L = L + L
#     print(L)


##############
## Clear a list and check that it's the same object
################
# L = [4,5,6]
# print(L)
# print(id(L))
# L.clear()
# print(L)
# print(id(L)) # same as 3 lines up, same object

## vs.

# L = [4,5,6]
# print(L)
# print(id(L))
# L = []
# print(L)
# print(id(L))  # different than 3 lines up, different object!


#######################################
############# ANSWERS TO YOU TRY IT ##########################
#######################################
def make_ordered_list(n):
    """ n is a positive int
    Returns a list containing all ints in order 
    from 0 to n (inclusive)
    """
    L = []
    for i in range(n+1):
        L.append(i)
    return L

# print(make_ordered_list(6))  # prints [0, 1, 2, 3, 4, 5, 6]

def remove_elem(L, e):
    """ 
    L is a list
    Returns a list with elements in the same order as L
    but without any elements equal to e. 
    """
    Lout = []
    for i in L:
        if e != i:
            Lout.append(i)
    return Lout

# L = [1,2,2,2]
# print(remove_elem(L, 1))    # prints [2,2,2]
# L = [1,2,2,2]
# print(remove_elem(L, 2))    # prints [1]
# L = [1,2,2,2]
# print(remove_elem(L, 0))    # prints [1,2,2,2]

def count_words(s):
    """ s is a string representing a sentence 
    Returns how many words are in s (i.e. a word is a 
    a sequence of characters between spaces. """
    words = s.split(' ')
    return len(words)

# s = "Hello it's me"
# print(count_words(s))   # prints 3

# s = "I just took a DNA test turns out I'm 100% splitting strings"
# print(count_words(s))   # prints 12

def sort_words(s):
    """ s is a string representing a sentence 
    Returns a list containing all the words in s but
    sorted in alphabetical order. """
    words = s.split(' ')
    # one way
    return sorted(words)
    # another way
    words.sort()
    return words

# s = "look at this photograph"
# print(sort_words(s))    # prints ['at', 'look', 'photograph', 'this']

# s = "now this is a story all about how my life got flipped turned upside down"
# print(sort_words(s))


#######################################
############ AT HOME ###############
#######################################
## Question 1
# L1 = ['re']
# L2 = ['mi']
# L3 = ['do']
# L4 = L1 + L2
# L3.extend(L4)
# L3.sort()
# del(L3[0])
# L3.append(['fa', 'la'])
# What's the value of L3 here?

## Question 2
# L1 = ['bacon', 'eggs']
# L2 = ['toast', 'jam']
# brunch = L1
# L1.append('juice')
# brunch.extend(L2)
# What's the value of brunch here?

## Question 3. 
def apply_to_each(L, f):
    """ L is a list of numbers 
        f is a list that takes in a number and returns a number
    Mutate L such that you apply function f to every element in L """
    # your code here

# test = [1,-2,3]
# apply_to_each(test, lambda x: x**2)
# print(test)   # prints [1,4,9]

# test = [-7, 8, 5, -8, -3]
# apply_to_each(test, abs)
# print(test)   # prints [7, 8, 5, 8, 3]



########################################
########## ANSWERS TO AT HOME ##############################
########################################

def apply_to_each(L, f):
    """ L is a list of numbers 
        f is a list that takes in a number and returns a number
    Mutate L such that you apply function f to every element in L """
    for i in range(len(L)): 
        L[i] = f(L[i])

# test = [1,-2,3]
# apply_to_each(test, lambda x: x**2)
# print(test)   # prints [1,4,9]

# test = [-7, 8, 5, -8, -3]
# apply_to_each(test, abs)
# print(test)   # prints [7, 8, 5, 8, 3]


############################
############################

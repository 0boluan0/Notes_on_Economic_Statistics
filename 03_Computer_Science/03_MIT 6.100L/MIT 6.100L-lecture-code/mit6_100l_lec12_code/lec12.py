## List comprehensions

L = [e**2 for e in range(6)] 	
# print(L)
L = [e**2 for e in range(8) if e%2 == 0] 
# print(L)
L = [[e,e**2] for e in range(4) if e%2 != 0] 
# print(L)

# Equivalent function to a list comprehension
# Notice how verbose this is!!
def f(expr, old_list, test = lambda x: True):
    new_list = []
    for e in old_list:
        if test(e):
            new_list.append(expr(e))
    return new_list

########## YOU TRY IT #############
## What is returned by this list comprehension expression?
# L = [x**2 for x in [2, 'a', 3, 4.0] if type(x) == int] 
# print(L)

###################################



## Keyword arguments aka default parameters
#########################
### EXAMPLE: bisection square root as a function (from lec 7)
#########################
def bisection_root(x):
    epsilon = 0.01
    low = 0
    high = x
    guess = (high + low)/2.0
    while abs(guess**2 - x) >= epsilon:
        if guess**2 < x: 
            low = guess
        else: 
            high = guess
        guess = (high + low)/2.0
    return guess


# print(bisection_root(4))
# print(bisection_root(123))

#########################
### EXAMPLE: improved bisection square root as a function
# takes in x and an epsilon
#########################
def bisection_root_new(x, epsilon):
    num_guesses = 0
    low = 0
    high = x
    guess = (high + low)/2.0
    while abs(guess**2 - x) >= epsilon:
        if guess**2 < x: 
            low = guess
        else: 
            high = guess
        guess = (high + low)/2.0
        num_guesses += 1
    print('num_guesses =', num_guesses)
    return guess

# print(bisection_root_new(123, 0.1))
# print(bisection_root_new(123, 0.00001))


#########################
### EXAMPLE: improved bisection square root as a function
# takes in x and an epsilon as a default parameter
#########################
def bisection_root_new(x, epsilon=0.01):
    num_guesses = 0
    low = 0
    high = x
    guess = (high + low)/2.0
    while abs(guess**2 - x) >= epsilon:
        if guess**2 < x: 
            low = guess
        else: 
            high = guess
        guess = (high + low)/2.0
        num_guesses += 1
    print('num_guesses =', num_guesses)
    return guess

# print(bisection_root_new(123))
# print(bisection_root_new(123, 0.5))
# print(bisection_root_new(123, epsilon=0.00001))
# print(bisection_root_new(epsilon=0.001, x=123)) 


#################
## EXAMPLE: function returning a function
#################
def make_prod(a):
    def g(b):
        return a*b
    return g

# # call it this way 1
# val = make_prod(2)(3)
# print(val)

# # call it this way 2
# doubler = make_prod(2)
# val = doubler(3)
# print(val)


#######################################
##### STEPS TO DEBUG THE FOLLOWING BUGGY CODE ########
#######################################
## STEP 1: run it with test cases
def is_pal(x):
    """ Returns True is list x is a palindrome and False otherwise """
    temp = x
    temp.reverse
    if temp == x:
        return True
    else:
        return False

# print(is_pal(list('abcba')))  # input is ['a','b','c','b','a']
# print(is_pal(list('ab')))     # input is ['a','b']

## STEP 2: add print statements about halfway through       
def is_pal(x):
    """ Returns True is list x is a palindrome and False otherwise """
    temp = x
    temp.reverse
    print(temp, x)  # add this
    if temp == x:
        return True
    else:
        return False

# print(is_pal(list('abcba')))  # input is ['a','b','c','b','a']
# print(is_pal(list('ab')))     # input is ['a','b']
 
## STEP 3: Add more print statements before and after critical points
def is_pal(x):
    """ Returns True is list x is a palindrome and False otherwise """
    temp = x
    print('before reverse', temp, x)  # add this
    temp.reverse
    print('after reverse', temp, x)  # add this
    if temp == x:
        return True
    else:
        return False
 
# print(is_pal(list('abcba')))  # input is ['a','b','c','b','a']
# print(is_pal(list('ab')))     # input is ['a','b']

## STEP 4: Fix one issue, notice something is still wrong
def is_pal(x):
    """ Returns True is list x is a palindrome and False otherwise """
    temp = x
    print('before reverse', temp, x)
    temp.reverse()                      # fix this
    print('after reverse', temp, x)
    if temp == x:
        return True
    else:
        return False

# print(is_pal(list('abcba')))  # input is ['a','b','c','b','a']
# print(is_pal(list('ab')))     # input is ['a','b']

## STEP 5: Recall with lists, aliasing/mutability is an issue
def is_pal(x):
    """ Returns True is list x is a palindrome and False otherwise """
    temp = x[:]     # fix this
    print('before reverse', temp, x)
    temp.reverse()
    print('after reverse', temp, x)
    if temp == x:
        return True
    else:
        return False

# print(is_pal(list('abcba')))  # input is ['a','b','c','b','a']
# print(is_pal(list('ab')))     # input is ['a','b']


########## YOU TRY IT AT HOME #############
## Write a list comprehension expression that uses a list named L.
# It makes a new list whose elements are the middle 
# character of strings whose length is 3. 

# If L = ['abc', 'm', 'p', 'xyz', '123', 57]
# It makes ['b', 'y', '2']
L = ['abc', 'm', 'p', 'xyz', '123', 57]


## There is a file lec12_wordle.py that is buggy!
## Try to fix the code to play the game correctly
###################################


########## ANSWERS TO YOU TRY IT AT HOME #############
## Write a list comprehension expression that uses a list named L.
# It makes a new list whose elements are the middle 
# character of strings whose length is 3. 

# If L = ['abc', 'm', 'p', 'xyz', '123']
# It makes ['b', 'y', '2']
L = ['abc', 'm', 'p', 'xyz', '123', 57]
# print([e[1] for e in L if len(e)==3 and type(e)==str])
###################################





## multiplying a*b using a for loop

def mult(a, b):
    total = 0
    for n in range(b):
        total += a
    return total

# print(mult(5,4))


####################################

## a*b using a while loop
def mult_iter(a, b):
    result = 0
    while b > 0:
        result += a
        b -= 1
    return result

# print(mult_iter(5,4))

## a*b recursive 
def mult_recur(a, b):
    if b == 1:
        return a
    else:
        return a + mult_recur(a, b-1)

# print(mult_recur(5,4))

def mult_recur_verbose(a, b):
    if b == 1:
        print("call with",a,b)
        return a
    else:
        print("call with",a,b)
        calc = mult_recur_verbose(a, b-1)
        print(f"returning {a}+{calc} for call with {a} and {b}")
        return a + calc

# print(mult_recur_verbose(5,4))


############### YOU TRY IT #################
# Calculate n**p recursively by writing this function
def power_recur(n, p):
    if ____ :
        return ____
    elif ____ :
        return ____
    else:
        return ____

# print(power_recur(2,3))  # prints 8

################################################
    
## factorial iterative
def factorial_iter(n):
    print(f'this is fact({n})')
    prod = 1
    for i in range(1,n+1):
        prod *= i
    return prod

# factorial_iter(5)

## factorial recursive
def fact_recur(n):
    print(f'this is fact({n})')
    if n == 1:
        return 1
    else:
        return n*fact_recur(n-1)

# fact_recur(5)


####################################################
############### ANSWERS TO YOU TRY IT #################
####################################################
# Calculate n**p recursively by writing this function
def power_recur(n, p):
    if p==0:
        return 1
    elif p==1:
        return n
    else:
        return n*power_recur(n,p-1)

# print(power_recur(2,3))   # prints 8


#######################################################
################ AT HOME ##############################
#######################################################
# Q1. Rewrite this to calculate b+b+b... a times
def mult(a, b):
    # your code here
    pass
    
# print(mult(5,4))

# Q2. If we evaluate mult_recur(3,4), how many times is the 
# procedure mult_recur called (including initial call)?
# Hint: add a print inside the function!

def mult_recur(a,b):
    if b == 0:
        return 0
    else:
        return a + mult_recur(a,b-1)

# mult_recur(3,4)


# Q3. Calculate a+b recursively. Assume the only math operation
# you are allowed to do are adding and subtracting 1
def add(a, b):
    """ Uses recursion to calculate a+b as adding
    a to 1, b times. """
    # your code here
    
# print(add(3,4))   # prints 7


# Q4. Calculate a+b recursively by 1's. Assume the only math operation
# you are allowed to do are adding and subtracting 1
def add_by_ones(a, b):
    """ Uses recursion to calculate a+b as adding
    1, a times then adding 1, b times. """
    # your code here

# print(add_by_ones(3,4))   # prints 7




#######################################################
################ ANSWERS TO AT HOME ###################
#######################################################
# Q1. Rewrite this to calculate b+b+b... a times
def mult(a, b):
    total = 0
    for n in range(a):
        total += b
    return total
    
# print(mult(5,4))

# Q2. If we evaluate mult_recur(3,4), how many times is the 
# procedure mult_recur called (including initial call)?
# Hint: add a print inside the function!
def mult_recur(a,b):
    print('inside')  # add this!
    if b == 0:
        return 0
    else:
        return a + mult_recur(a,b-1)

# mult_recur(3,4)

# Q3. Calculate a+b recursively. Assume the only math operation
# you are allowed to do are adding and subtracting 1
def add(a, b):
    """ Uses recursion to calculate a+b as adding
    a to 1, b times. """
    # base case
    if b == 0:
        return a
    else:
    # recursive case
        return 1+add(a,b-1)
    
# print(add(3,4))   # prints 7


# Q4. Calculate a+b recursively by 1's. Assume the only math operation
# you are allowed to do are adding and subtracting 1
def add_by_ones(a, b):
    """ Uses recursion to calculate a+b as adding
    1, a times then adding 1, b times. """
    # base case
    if b == 0:
        return a
    if a == 0:
        return b
    else:
    # recursive case
        return add(1,b-1)+add(a-1,1)

# print(add_by_ones(3,4))   # prints 7




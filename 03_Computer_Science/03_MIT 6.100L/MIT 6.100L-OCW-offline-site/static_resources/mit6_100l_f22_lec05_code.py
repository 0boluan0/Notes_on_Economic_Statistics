#################
## EXAMPLE: successive addition
#################

# 0.125 is a perfect power of 2
# x = 0
# for i in range(10):
#     x += 0.125
# print(x == 1.25)

#######

# 0.1 is not a perfect power of 2
# x = 0
# for i in range(10):
#     x += 0.1
# # print(x == 1)

# print(x, '==', 10*0.1)

#############
## EXAMPLE
# protip: use Python Tutor to go step-by-step: http://pythontutor.com/
#############

# x = float(input('Enter a decimal number between 0 and 1: '))

# p = 0
# while ((2**p)*x)%1 != 0:
#     print(f'Remainder = {str((2**p)*x - int((2**p)*x))}')
#     p += 1

# num = int(x*(2**p))

# result = ''
# if num == 0:
#     result = '0'
# while num > 0:
#     result = str(num%2) + result
#     num = num//2

# for i in range(p - len(result)):
#     result = '0' + result

# result = result[0:-p] + '.' + result[-p:]
# print(f'The binary representation of the decimal {str(x)} is {str(result)}')


################
## EXAMPLE: Approximation by epsilon increments
## Incrementally fixing code as we find issues with approximation
################

# try with 36, 24, 2, 12345
# x = 36
# epsilon = 0.01
# num_guesses = 0
# guess = 0.0
# increment = 0.0001
# while abs(guess**2 - x) >= epsilon:
#     guess += increment
#     num_guesses += 1
# print(f'num_guesses = {num_guesses}')
# print(f'{guess} is close to square root of {x}')

###########

# Caution, you'll need to "Restart Kernel" in the shell if you run this code
# x = 54321
# epsilon = 0.01
# num_guesses = 0
# guess = 0.0
# increment = 0.0001
# while abs(guess**2 - x) >= epsilon:
#     guess += increment
#     num_guesses += 1
#     if num_guesses%100000 == 0:
#         print(f'Current guess = {guess}')
#         print(f'Current guess**2 - x = {abs(guess*guess - x)}')
#     if num_guesses%1000000 == 0:
#         input('continue?')
# print(f'num_guesses = {num_guesses}')
# print(f'{guess} is close to square root of {x}')

##########

# Add an extra stopping condition 
# and check for why the loop terminated
# x = 54321
# epsilon = 0.01
# num_guesses = 0
# guess = 0.0
# increment = 0.0001  # try with 0.00001
# while abs(guess**2 - x) >= epsilon and guess**2 <= x:
#     guess += increment
#     num_guesses += 1
# print(f'num_guesses = {num_guesses}')
# if abs(guess**2 - x) >= epsilon:
#     print(f'Failed on square root of {x}')
#     print(f'Last guess was {guess}')
#     print(f'Last guess squared is {guess*guess}')
# else:
#     print(f'{guess} is close to square root of {x}')
    
#######


#################################################
######################## AT HOME ##########################
#################################################
# 1. If you are incrementing from 0 by 0.022, how many increments 
# can you do before you get a floating point error? 

# x = 0
# count = 20     # check different numbers here
# for i in range(count):
#     x += 0.022 # increment
#     print(x)      # check this value for floating point error


# 2. Automate the code from the previous problem. Suppose you are 
# just given an increment value. Write code that automatically
# determines how many times you can add increment to itself 
# until you start to get a floating point error.

# your code here

#################################################
#################################################
#################################################


#################################################
################ ANSWER TO AT HOME ##########################
#################################################
# Automate the code. Suppose you are 
# just given an increment value. Write code that automatically
# determines how many times you can add increment to itself 
# until you start to get a floating point error.

# n = 0.022
# N = 1
# x = n
# while x == n*N:
#     print(x)
#     x += n
#     N += 1
# note that the x and N increments one extra time 
# print(f'count is {N-1} where {x-n} != {n*(N-1)}')

#################################################
#################################################
#################################################

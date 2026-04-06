###################
# Tou can uncomment each of these examples
# and try running them yourself

# To batch comment/uncomment, select the lines and then
# on Windows hit CTRL+1 or on Mac hit CMD+1
###################



###################
# EXAMPLE: while loops 
####################
# where = input("You are in the Lost Forest. Go left or right? ")
# while where == "right":
#     where = input("You are in the Lost Forest. Go left or right? ")
# print("You got out of the Lost Forest! \o/")



###########################################

# Fun Lost Forest code, run it on your own!
#where = input("You are in the Lost Forest\n****************\n****************\n :)\n****************\n****************\nGo left or right? ")
#while where.lower() == "right":
#    where = input("You are in the Lost Forest\n****************\n******       ***\n  (╯°□°）╯\n     ︵ \n    ┻━┻\n****************\n****************\nGo left or right? ")
#print("\nYou got out of the Lost Forest!\n\o/")

    
###########
## EXAMPLE    
###########
# n = int(input('Please enter a non-negative integer: '))
# while n > 0:
#     print('x')
#     n = n-1  # the same as n -= 1
    

################ YOU TRY IT ###################
## EXAMPLE: infinite loop, be careful!
# To stop it, click the shell and hit CTRL+c or 
# the red square at the top of the shell
##############################################
# while True:
#     print("noooooooo")



############### YOU TRY IT ################
# Expand this code to show a sad face when the user entered 
# the while loop more than 2 times. Hint: use a counter
###################
# where = input("Go left or right? ")
# while where == "right":
#     where = input("Go left or right? ")
# print("You got out!")



#############
## EXAMPLE: counter
#############

## With while loop
# n = 0
# while n < 5:
#     print(n)
#     n = n+1

## With for loop
#for n in range(5):
#    print(n)

###########
## EXAMPLE: factorial
###########

## With while loops
# x = 6
# i = 1
# factorial = 1
# while i <= x:
#     factorial *= i
#     i += 1
# print(f'{x} factorial is {factorial}')

## With for loops
# factorial = 1
# for i in range(1, x+1, 1):
#     factorial *= i
# print(f'{x} factorial is {factorial}')


################ YOU TRY IT ################
# for i in range(1,4,1):
#     print(i)
# for j in range(1,4,2):
#     print(j*2)
# for me in range(4,0,-1):
#     print("$"*me)


###########################################

###############
## EXAMPLE: sum
###############

#mysum = 0
#for i in range(10):
#    mysum += i
#print(mysum)

######

#mysum = 0
#for i in range(7, 10):
#    mysum += i
#print(mysum)

######

#mysum = 0
#for i in range(5, 11, 2):
#    mysum += i
#    if mysum == 5:
#        break
#        mysum += 1
#print(mysum)

################ YOU TRY IT ################
# Fix this code to use variables start and end in the 
# range, to get the total sum between and including those values. 

# mysum = 0
# start = 3
# end = 5
# for i in range(start, end):
#     mysum += i
# print(mysum)

###########################################



#########################################################
##################### AT HOME ###########################
#########################################################

# Practice 1: 
# Declare a variable x that stores an int > 0. Print all ints, one on each
# line, between 1 (inclusive) and x (inclusive) that are divisible by 5.
# For ex. if x = 15, it prints 5, 10, and 15. 
# For ex. if x = 14, it prints 5 and 10.


# Practice 2:
# Declare a variable n that stores an int. Print the sum of all digits 
# in n. Hint: you can get a digit at a time looking at the remainder 
# when you divide n by 10.
# For ex. If x = 1234, print 10
 



#########################################################
##################### END AT HOME ###########################
#########################################################


#########################################################
##################### ANSWERS AT HOME ###########################
#########################################################

# Practice 1: 
# Declare a variable x that stores an int > 0. Print all ints, one on each
# line, between 1 (inclusive) and x (inclusive) that are divisible by 5.
# For ex. if x = 15, it prints 5, 10, and 15. If x = 14, it prints 5 and 10.

# x = 15
# for i in range(1,x+1):
#     if i%5 == 0:
#         print(i)


# Practice 2:
# Declare a variable n that stores an int. Print the sum of all digits 
# in n. Hint: you can get a digit at a time looking at the remainder 
# when you divide n by 10.
# For ex. If x = 1234, print 10
# n = 1234
# total = 0
# while True:
#     r = n%10
#     total += r 
#     n = n//10
#     if n == 0:
#         break
# print(total)

#########################################################
##################### END ANSWERS AT HOME ###########################
#########################################################




#########################################
############### ANSWERS TO LECTURE ##########################
#########################################
# You Try It 1: 
# Expand this code to show a sad face when the user entered 
# the while loop more than 2 times. Hint: use a counter
###################
# where = input("Go left or right? ")
# counter = 0
# while where == "right":
#     counter = counter + 1
#     if counter > 2:
#         print(":(")
#     where = input("Go left or right? ")
# print("You got out!")



# Your Try It 2: 
# Fix this code to use variables start and end in the 
# range, to get the total sum between and including those values. 

# mysum = 0
# start = 1
# end = 3
# for i in range(start, end+1):
#     mysum += i
# print(mysum)

#########################################
############### END ANSWERS TO LECTURE ##########################
#########################################

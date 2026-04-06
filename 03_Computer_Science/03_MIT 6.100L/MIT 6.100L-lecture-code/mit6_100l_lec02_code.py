
## TYPE THIS IN THE CONSOLE -- STRINGS ##
a = 'me'
b = "myself"
c = a + b
d = a + " " + b
silly = a * 3

s = "abc"
len(s) 

## TYPE THIS IN THE CONSOLE -- INDEXING ##
s = "abc"
s[0]
s[1]
s[2]
#s[3]  # this is an error
s[-1]
s[-2]
s[-3]

## TYPE THIS IN THE CONSOLE -- SLICING ##
s = "abcdefgh"
s[3:6]
s[3:6:2] 
s[:]
s[::-1] 
s[4:1:-2]

## TYPE THIS IN THE CONSOLE - MANIPULATION ##
s = "car"
#s[0] = 'b'  # this is an error
s = 'b'+s[1:len(s)] 

#########################################
############### LECTURE ##########################
#########################################

# ## PRINTING ##
a = "the"
b = 3
c = "musketeers"
# print(a, b, c)
# print(a + b + c)   # this is an error
# print(a + str(b) + c)

# num = 5
# print("my num is", num)
# s = "my num is" + str(num)
# print(s)

# x = 1
# x_str = str(x)
# print("my fav num is", x, ".", "x =", x)
# print("my fav num is " + x_str + ". " + "x = " + x_str)

# ## USER INPUT ##
# #Example 1
# text = input("Type anything... ")
# print(5*text)

# #Example 2
# num1 = input("Type a number: ")
# print(5*num1)
# num2 = int(input("Type a number: "))
# print(5*num2)

############## YOU TRY IT ###############
# Write a program that: 
# * Asks the user for a verb.
# * Prints "I can _ better than you" where you replace _ with the verb.
# * Then prints the verb 5 times in a row separated by spaces.
# For example, if the user enters run, you print:
#     I can run better than you!
#     run run run run run

# your code here



#########################################

# #Example 3 - Newton's Method for cube root
# x = int(input('What x to find the cube root of? '))
# g = int(input('What guess to start with? '))

# print('Current estimate cubed = ', g**3)
# next_g = g - ((g**3 - x)/(3*g**2))
# print('Next guess to try = ', next_g)


# ## F-STRINGS ##
# num = 3000
# fraction = 1/3
# print(num*fraction, 'is', fraction*100, '% of', num)
# print(num*fraction, 'is', str(fraction*100) + '% of', num)
# print(f'{num*fraction} is {fraction*100}% of {num}')

# print(f'{num*fraction:,.0f} is {fraction*100:,.2f}% of {num:,}')


# pset_time = 15
# sleep_time = 8
# print(sleep_time > pset_time)
# derive = True
# drink = False
# both = drink and derive
# print(both)


############## YOU TRY IT ###############
# Write a program that:
# * Saves a secret number. 
# * Asks the user for a number guess.
# * Prints a bool depending on whether the guess matches the secret.

# your code here

    

#########################################

# ## BRANCHING ##
# #Example 1
# pset_time = 22
# sleep_time = 8
# if (pset_time + sleep_time) > 24:
#     print("impossible!")
# elif (pset_time + sleep_time) >= 24:
#     print("full schedule!")
# else:
#     leftover = abs(24-pset_time-sleep_time)
#     print(leftover,"h of free time!")
# print("end of day")


############## YOU TRY IT ###############
# # Buggy, fix it!
# x = int(input("Enter a number for x: "))
# y = int(input("Enter a different number for y: "))
# if x == y:
#     print(x,"is the same as",y)
# print("These are equal!")

#########################################

# ## NESTED BRANCHING ##
# #Example 1
# x = float(input("Enter a number for x: "))
# y = float(input("Enter a number for y: "))
# if x == y:
#     print("x and y are equal")
#     if y != 0:
#         print("therefore, x / y is", x/y)
# elif x < y:
#     print("x is smaller")
# else:
#     print("y is smaller")
# print("thanks!")


############## YOU TRY IT ###############
# What's printed when y = 2, y = 20, y = 11?
# What if "if x <= y:" becomes "elif x <= y:"

# answer = ''
# x = 11
# y = 2 # try 20 and 11
# if x == y:
#     answer = answer + 'M'
# if x <= y:   # try making this line: elif x <= y:
#     answer = answer + 'i'
# else:
#     answer = answer + 'T'
# print(answer)

#########################################

############## YOU TRY IT ###############
# Write a program that:
# * Saves a secret number. 
# * Asks the user for a number guess.
# * Prints whether the guess is too low, too high, or the same as the secret. 

# your code here

#############################################

#########################################
############### END LECTURE ##########################
#########################################



#########################################
############### AT HOME ###################
#########################################
# Practice 1: What is the value of s1 and s2?
s1 = "a" + "b"

d = "hi"
e = " ana"
s2 = d + 2*e


# Practice 2: What are the substrings of s?
s = "ABC d3f ghi"
s[0:3:1]
s[0:4]
s[8:len(s):3]
s[2::-1]


# Practice 3: What does this print?
# Note that a += b is the same as a = a + b
# answer = ''
# x = 11
# # try with y = 2 and y = 12
# y = 2
# if len(str(x)) == len(str(y)):
#     if y != 0 and x%2 == 1:
#         answer = answer + "x / y is " + str(x/y)
# elif x < y:
#     answer += "\nx is smaller"  # \n inserts a newline character in the string
# else:
#     answer += "\ny is smaller"
# print(answer)



# Practice 4: Uncomment the code below and:
# What does it print when a = 6 and b = "6"
# What does it print when a = "1" and b = 1
# What does it print when a = 3 and b = 3
# What does it print when a = "1" and b = "1"

# if ( a == int(b) ):
#     print("int conversion")
# if ( a == int(b) ) and ( str(a) == b ):
#     print("int and str conversion")
# else: 
#     print("interesting")
    
#########################################
############### END AT HOME ##########################
#########################################


#########################################
############### ANSWERS TO LECTURE ##########################
#########################################
# You Try It 1: Write a program that: 
# * Asks the user for a verb.
# * Prints "I can _ better than you" where you replace _ with the verb.
# * Then prints the verb 5 times in a row separated by spaces.
# For example, if the user enters run, you print:
#     I can run better than you!
#     run run run run run

# your code here
# verb = input("Type in a verb: ")
# print("I can", verb, "better than you!")
# print((verb+" ")*4+verb)


# You Try It 2: Write a program that:
# * Saves a secret number. 
# * Asks the user for a number guess.
# * Prints a bool depending on whether the guess matches the secret.

# your code here
# secret = 7
# guess = int(input("Guess a number between 0 and 10: "))
# print(secret == guess)


# You Try It 3: Buggy, fix it!
# x = int(input("Enter a number for x: "))
# y = int(input("Enter a different number for y: "))
# if x == y:
#     print(x,"is the same as",y)
# print("These are equal!")

# Fixed:
# x = int(input("Enter a number for x: "))
# y = int(input("Enter a different number for y: "))
# if x == y:
#     print(x,"and",y)
#     print("These are equal!")


# You Try It 4: Write a program that:
# * Saves a secret number. 
# * Asks the user for a number guess.
# * Prints whether the guess is too low, too high, or the same as the secret. 

# your code here
# secret = 7
# guess = int(input("Guess a number between 0 and 10: "))
# if guess == secret:
#     print("You are correct.")
# elif guess < secret:
#     print("Your guess is too low.")
# else:
#     print("Your guess is too high.")

#########################################
############### END ANSWERS TO LECTURE ##########################
#########################################

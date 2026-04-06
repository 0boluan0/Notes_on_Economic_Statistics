####################################################################################
# Practice Problem 1
# Write a program that takes your name as an input and outputs the length of your name minus 5.

# Insert code below
name = input("Enter name: ")
print(len(name) - 5)


####################################################################################
# Practice Problem 2
# Write a program to remove the nth character from a non empty string. 
# Print the old string and the new string.

test_string = "We want to remove the nth character from this string"  # define non-empty string
n = 8  # we set n to be 8

# Insert code below
new_string = test_string[:n-1] + test_string[n:]
print(test_string)
print(new_string)


####################################################################################
# Practice Problem 3
# Write a program which answers the following:
# Does a given string have length greater than 10 or less than 5? If True, output True. If False, output False.

my_string = "This is my string"  # example string - modify to test

# Insert code below
if len(my_string) > 10 or len(my_string) < 5:
    print(True)
else:
    print(False)



# Practice Problem 4
# Write a program which answers the following using a for loop:
# Count the number of e's in the following string

my_string = "How many times is the letter e in this string?"

# Insert code below
number_of_es = 0  # this will be our counter

for letter in my_string:
    if letter == "e":
        number_of_es += 1

print(number_of_es)





























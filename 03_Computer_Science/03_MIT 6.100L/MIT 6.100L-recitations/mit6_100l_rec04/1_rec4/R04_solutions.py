# Problem 1: Lamba Functions Practice
# a) Write a lambda function that calculates the cube root of a given number 
# passed in as an argument
f1 = lambda x: x ** (1/3)

# b) Write a lambda function that takes in two arguments and outputs the product
# of those two numbers. 
f2 = lambda x, y: x * y

# test
# print(f1(8))
# print(f1(4))
# print(f2(1,2))
# print(f2(4,5))


#############################################################################
# Problem 2: Practice working with Tuples:
# Write a function that counts the number of times the number 1 appears 
# in an inputted tuple.

def count_number_one(tup):
    count = 0
    for elem in tup:
        if elem == 1:
            count += 1
    return count

# print(count_number_one((1,2,3,4,5,1,1)))


#############################################################################
# Problem 3: Practice working with Python Tuples
# Write a Function that takes in two tuples and outputs a single tuple containing 
# only common elements of both tuples. 
def common_elements(tup1, tup2):
    output_tuple = ()
    for x in tup1:
        for y in tup2:
            if x == y:
                output_tuple += (x,)
    return output_tuple

# test
# print(common_elements((2,3,4), (3,4,5,6)))



#############################################################################
# Problem 4: Practice working with Python Lists
# Write a Python program to remove sublists from a given list of lists, which 
# contain an element outside a given range.
# e.g 
# Original list:
# [[2], [0], [1, 2, 3], [0, 1, 2, 3, 6, 7], [9, 11], [13, 14, 15, 17]]
# After removing sublists from a given list of lists, which contain an 
# element outside the given range of 12 - 20 (inclusive):
# [[13, 14, 15, 17]]

def remove_list_range(input_list, low, high):
    output_list = []
    for i in input_list:
        if min(i) >= low and max(i) <= high:
            output_list.append(i)
    return output_list

# test 
# print(remove_list_range([[2], [0], [1, 2, 3], [0, 1, 2, 3, 6, 7], [9, 11], [13, 14, 15, 17]], 13, 17))


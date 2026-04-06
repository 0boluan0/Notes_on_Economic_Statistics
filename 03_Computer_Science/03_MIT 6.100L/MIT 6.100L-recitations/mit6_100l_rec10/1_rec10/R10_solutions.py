#############################################################################
# Problem 1:
# For each of the following expressions, what is the order of growth class 
# that best describes it?
# a) 8n   
# b) 3n**2 + 7n**3 + 4   
# c) 5log(n) + 5n
# d) 3**n + n**2
# e) 20n + nlog(n)
# f) 5 + 60 
# g) log(n) + 4n


# Answers: 
# a) O(n)
# b) O(n**3)
# c) O(n)
# d) O(3**n)
# e) O(nlog(n))
# f) O(1)
# g) O(n)


#############################################################################
# Problem 2:
# Rank the following functions by runtime complexity. Note some may have 
# the same runtime complexity.

# f(n) = n**2 + 4n + 2
# g(n) = log(n**2)
# h(n) = log2(n)  (i.e. log base 2 n)
# j(n) = 3n**3 + 2
# l(n) = n!
# k(n) = 2**n

# Answer: 
# Ranked from fastest to slowest:
# g(n) = h(n), f(n), j(n), k(n), l(n)



#############################################################################
# Problem 3:
# What is the time complexity for the following programs? 

def program1():
    my_list = [1,2,3,4,5,6,7,8]
    my_list_even= []
    for i in range(len(my_list)):
        if i % 2 == 0:
            my_list_even.append(i)
    return my_list

# Answer: 
# O(len(my_list)) -> if we assume len(my_list) = n, this gives O(n) 



def program2():
    my_list = [1,2,3,4,5,7]
    my_second_list = [1,2,3,4,5,7]
    
    output_list1 = [i for i in my_list]

    output_list2 = []
    for i in my_list:
        for j in my_second_list:
            output_list2 += [i,j]
    
    return (output_list1, output_list2)


# Answer: Ignoring operations which are O(1) in the program, we are left 
# with two for loops. The first is O(n) and the second is O(n**2). 
# So we have O(n + n**2) = O(n**2)
    

def program3(n):
    epsilon = 0.01
    low = 0
    high = n
    ans = (high + low) / 2 
    
    while abs(ans**4 - n) >= epsilon:
        if ans**4 > n:
            high = ans 
        else: 
            low = ans
        ans = (high + low) / 2
    return ans



# Answer: 
# This is a binary search algorithm so we would get logarithmic time 
# complexity of O(log(n))


#############################################################################
# Problem 4:
# Describe two ways to construct an algorithm to find the maximum number is a list. 
# One algorithm should have time complexity O(n), the other O(n**2). (Note the 
# O(n**2) algorithm is highly inefficient and we'd never actually use it 
# in practice).

# Answer:
# The first function should loop through the list, comparing each element to 
# the maximum element seen so far. We loop through the list only once giving 
# us O(n) time complexity.  
        
# The second function would compare each number to every other number on the list. 
# This could be a achieved by using a nested for loop where we loop over the list twice, giving
# O(n**2) time complexity. 



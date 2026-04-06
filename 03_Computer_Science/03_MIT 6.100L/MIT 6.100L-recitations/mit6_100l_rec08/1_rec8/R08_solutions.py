# RECURSSION PRACTICE

# Problem 1: Write a recursive program to calculate the sum of the positive 
# integers of n+(n-2)+(n-4)... (until and not including n-x =< 0)
def sum_series(n):
    if n <=0:
        return 0
    else:
        return n + sum_series(n - 2)

# testing
print(sum_series(6))  # 12 
print(sum_series(10))  # 30 
print(sum_series(5))  # 9   


# Problem 2: Write a recursive program to calculate the value of 'a' to the power 'b'
def power(a, b):
    if b == 0:
        return 1
    else:
        return a * power(a, b-1)

# testing
print(power(4, 3))  # 64
print(power(3, 4))  # 81


# Problem 3: Write a recursive program to calculate the sum of a list of numbers.
def list_sum(num_List):
    if len(num_List) == 1:
        return num_List[0]
    else:
        return num_List[0] + list_sum(num_List[1:])
    
# testing 
print(list_sum([2, 4, 5, 6, 7]))  # expect 24



# Problem 4: Write a recurssive program to calculate the nth harmonic number
def harmonic(n):
    if n == 1:
        return 1
    else:
        return 1/n + harmonic(n-1)

print(harmonic(3)) # expect 1.83333333333
print(harmonic(5)) # expect 2.28333333333




# Extra - Problem 5: Write a recursive program to find the greatest common divisor (gcd) 
# of two integers. 
def gcd(a, b):
    if(b==0):
        return a
    else:
        return gcd(b,a%b)

# Alternative solution
def gcd(a, b):
    if a%b == 0:
        return b
    else:
        return gcd(b, a%b)


# testing   
print(gcd(5, 4))  # 1 
print(gcd(15, 12))  # 3
print(gcd(12, 12))  # 12


    
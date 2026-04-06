import time

## -------------------------------------------------- ##
## EXAMPLE: timing a program
## -------------------------------------------------- ##

# constant fcn
def c_to_f(c):
    return c*9.0/5 + 32

# linear fcn -- finds 0+1+2+...+x
def mysum(x):
    total = 0
    for i in range(x+1):
        total += i
    return total

# quadratic fcn -- finds n*n inefficiently
def square(n):
    sqsum = 0
    for i in range(n):
        for j in range(n):
            sqsum += 1
    return sqsum

# helper function to show timing
def time_wrapper(f, L):
    print('Timing', f.__name__)
    for i in L:
        t = time.time()
        f(i)
        dt = time.time()-t
        print (f"{f.__name__}({i}) took {dt} sec")

##creates a list [1, 10, 100, ...] to test different input sizes
L_N = [1]
for i in range(8):
    L_N.append(L_N[-1]*10)

## time each function
# time_wrapper(c_to_f,  L_N)
# time_wrapper(mysum, L_N)
# time_wrapper(square, L_N)  # caution this will take 500 sec, then 50000 sec


## -------------------------------------------------- ##
## EXAMPLE: counting the number of operations
## -------------------------------------------------- ##

## constant fcn with counting the number of ops
def c_to_f(c):
    counter = 3
    return (counter, c*9.0/5 + 32)

# linear fcn  with counting the number of ops
def mysum(x):
    counter = 1
    total = 0
    for i in range(x+1):
        counter += 3
        total += i
    return (counter, total)

# quadratic fcn  with counting the number of ops
def square(n):
    counter = 1
    mysum = 0
    for i in range(n):
        counter += 1
        for j in range(n):
            counter += 3
            mysum += 1
    return (counter, mysum)

# helper function to show number of operations
def count_wrapper(f, L):
    print('Counting', f.__name__)
    for i in L:
        counter = f(i)[0]
        if i == min(L):
            multiplier = 1.0
        else:
            multiplier = counter/float(prev)
        prev = counter
        print(f"{f.__name__}({i}): {counter} ops, {round(multiplier,5)} x more")


L1 = [100]
for i in range(5):
    L1.append(L1[-1]*10)

L2_a = [128, 256, 512, 1024, 2048, 4096, 8192]
L2_b = [1, 10, 100, 1000, 10000]
# count_wrapper(c_to_f, L1)
# count_wrapper(mysum, L1)
# count_wrapper(square, L2_a)
# count_wrapper(square, L2_b)

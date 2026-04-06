import time
import math

## -------------------------------------------------- ##
## EXAMPLE: timing a program
## -------------------------------------------------- ##

## define two functions
def convert_to_km(m):
    return m * 1.609

def compound(invest, interest, n_months):
    total=0
    for i in range(n_months):
       total = total * interest + invest
    return total

##creates a list [1, 10, 100, ...] to test different input sizes
L_N = [1]
for i in range(8):
    L_N.append(L_N[-1]*10)

## ------------------------------------------------------------
## EXAMPLE: Report timing and ops/sec for two functions 
## ------------------------------------------------------------

###### print time and ops/sec for constant fcn
# for N in L_N:
#     t = time.perf_counter()
#     km = convert_to_km(N)
#     dt = time.perf_counter()-t
#     print (f"convert_to_km({N}) took {dt:.2e} sec ({1/dt:,.2f}/sec)")
    
##### print time and ops/sec for linear fcn
## invest is my changing variable
# for N in L_N:
#     t = time.perf_counter()
#     money = compound(N, 1.05, 12)
#     dt = time.perf_counter()-t
#     print (f"compound({N}) took {dt:.2e} seconds ({1/dt:,.2f}/sec)")

# ## interest is my changing variable
# for N in L_N:
#     t = time.perf_counter()
#     money = compound(10.0, 1+1/N, 12)
#     dt = time.perf_counter()-t
#     print (f"compound({N}) took {dt:.2e} seconds ({1/dt:,.2f}/sec)")

# ## n_months is my changing variable
# for N in L_N:
#     t = time.perf_counter()
#     money = compound(10.0, 1.05, N)
#     dt = time.perf_counter()-t
#     print (f"compound({N}) took {dt:.2e} seconds ({1/dt:,.2f}/sec)")

## ------------------------------------------------------------
## EXAMPLE: Repeat timing for a few samples and report average 
### run the computation n_samples times for better measurement
## ------------------------------------------------------------

###### print avg time and ops/sec for constant fcn
n_samples = 50
# for N in L_N:
#     t = time.perf_counter()
#     for i in range(n_samples):
#         km = convert_to_km(N)
#     dt = (time.perf_counter()-t)/n_samples
#     print ("convert(", N, ") took ", dt, "seconds")

###### print time and ops/sec for linear fcn
# following code is longer and easier to measure (also slower) 
# so use smaller n_sample
n_samples = 5
# for N in L_N:
#     t = time.perf_counter()
#     for i in range(n_samples):  
#         money = compound(10.0, 1.05, N)
#     dt = (time.perf_counter()-t)/n_samples
#     print (f"compound({N}) took {dt:.2e} seconds ({1/dt:,.2f}/sec)")


 
## ------------------------------------------------------------
## EXAMPLE: Timing computations with list data structures 
## ------------------------------------------------------------

##### sum of elements in L
def sum_of(L):
    total = 0.0
    for elt in L:
        total = total + elt
    return total

L_N = [1]
for i in range(8):
    L_N.append(L_N[-1]*10)

# for N in L_N:
#     L = list(range(N)) # makes list of length N [0,1,2,3,...N]
#     t = time.perf_counter()
#     s = sum_of(L)
#     dt = time.perf_counter()-t
#     print (f"sum_of length({N}) took {dt:.2e} seconds ({1/dt:,.2f}/sec)")

###### is an element in L?

## brute force algorithm
def is_in(L, x):
    for elt in L:
        if elt==x: 
            return True
    return False

## bisection search algorithm
def binary_search(L, x):
    """ Returns True if x is in L. L must be sorted """
    lo = 0
    hi = len(L)
    while hi-lo > 1:
        mid = (hi+lo) // 2
        if L[mid] <= x:
            lo = mid
        else:
            hi = mid
    return L[lo] == x

L_N = [1]
for i in range(8):
    L_N.append(L_N[-1]*10)

prev_time_brute = None
prev_time_binary = None
prev_time_builtin = None

## loop runs each of is_in, binary_search, and in keyword
# for N in L_N:
#     L = list(range(N)) # list [0,1,2,...N-1]

#     if (N <= 100000000):
#     ## time using the brute force, is_in function to check if elem in L
#         t = time.perf_counter()
#         for x in [L[0], L[len(L)//2], L[-1]]:  # look for 0, N/2, N
#             my_bool = is_in(L, x)
#         t = (time.perf_counter()-t) / 3        # avg time to find 3 values
#         print (f"is_in({N}) took {t:.2e} seconds ({1/t:,.2f}/sec)")
#         if not N == L_N[0]:
#             print (f'    {t/prev_time_brute:.2f} times more than for 10 times fewer elements')
#         prev_time_brute = t
        
#     # time using the binary search function to check if elem in L
#     t = time.perf_counter()
#     for x in [L[0], L[len(L)//2], L[-1]]:
#             my_bool = binary_search(L, x)
#     t = (time.perf_counter()-t)/3  # avg time to find 3 values
#     print (f"binary({N}) took {t:.2e} seconds ({1/t:,.2f}/sec)")
#     if not N == L_N[0]:
#         print (f'    {t/prev_time_binary:.2f} times more than for 10 times fewer elements')
#     prev_time_binary = t

#     ## time using the built-in function to check if elem in L
#     t = time.perf_counter()
#     for x in [L[0], L[len(L)//2], L[-1]]:  # look for 0, N/2 and N
#         my_bool = x in L
#     t = (time.perf_counter()-t)/3          # avg time to find 3 values
#     print (f"builtin({N}) took {t:.2e} seconds ({1/t:,.2f}/sec)")
#     if not N == L_N[0]:
#         print (f'    {t/prev_time_builtin:.2f} times more than for 10 times fewer elements')
#     prev_time_builtin = t


#     print()
    

## ------------------------------------------------------------
## EXAMPLE: Timing computations involving nested loops 
## ------------------------------------------------------------

def diameter(L):
    """ Assumes that L is a list of pairs of Cartesian coordinates
    compares all pairs of points and returns the largest distance """
    farthest_dist = 0
    for i in range(len(L)):
        # next loop only goes from i+1 so that pairs are not compared twice
        for j in range(i+1, len(L)):
            p1 = L[i]
            p2 = L[j]
            dist = math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
            if dist > farthest_dist:
                farthest_dist = dist
    return farthest_dist

def create_list_of_2D_points(N):
    L = []
    for i in range(N):
        x = math.cos(i)*i
        y = math.sin(i)*i
        L.append((x, y))
    return L

L_N = [100, 200, 400, 800, 1600, 3200, 6400]

diameter_times = []
last = 0
# for N in L_N:
#     L = create_list_of_2D_points(N)
#     t = time.perf_counter()
#     d = diameter(L)
#     dt = time.perf_counter() - t
#     diameter_times.append(dt)
#     print ("diameter of", N, "points took", dt, "seconds")
#     if (last != 0):
#         print(f"    delta = {dt / last:.1f}x for 2x as many points")
#     last = dt


## ------------------------------------------------------------
## EXAMPLE: Timing computations involving binary numbers
## ------------------------------------------------------------

def all_binary_numbers(N):
    def helper (prefix, N):
        if N==0:
            return [prefix]
        return helper(prefix+'0', N-1) + helper(prefix+'1', N-1)                        
    return helper('', N)

# for N in range(5):
#     print (all_binary_numbers(N))

# L_N = [5, 10, 15, 20, 25]

# for N in L_N:
#     t = time.perf_counter()
#     abn = all_binary_numbers(N)
#     t = time.perf_counter()-t
#     print ('binary numbers of ', N, 'digits took ', t, 's')


## ------------------------------------------------------------
## EXAMPLE: Counting operations
## ------------------------------------------------------------

global count

# define functions
def is_in_counter(L, x):
    global count
    count += 1
    for elt in L:
        count += 2 # set elt, if == test
        if elt==x: return True
    return False

def binary_search_counter(L, x):
    """ returns True if x is in L
        L must be sorted """
    global count
    lo = 0
    hi = len(L)
    count += 3 #set lo, hi, len
    while hi-lo > 1:
        count += 2 #while test, -
        mid = (hi+lo) // 2
        count += 3 #+, //, assign mid
        if L[mid] <= x:
            lo = mid
        else:
            hi = mid
        count += 3 #access mid, if test, assign mid
    count += 3 #access lo, == test, return
    return L[lo] == x

########### counting ops for is_in
L_N = [1]
for i in range(8):
    L_N.append(L_N[-1]*10)

is_in_counts = []
# for N in L_N:
#     count = 0
#     L = range(N)
#     for x in [L[0], L[len(L)//2], L[-1]]:
#         my_bool = is_in_counter(L, x)
#     print ('is_in with', N, 'elements did', count, 'operations')
#     is_in_counts.append(count)
    
    
########### counting ops for binary_search
L_N = [1]
for i in range(10):
    L_N.append(L_N[-1]*10)

bin_search_counts = []
# for N in L_N:
#     count = 0
#     L = range(N)
#     for x in [L[0], L[len(L)//2], L[-1]]:
#         my_bool = binary_search_counter(L, x)
#     print ('binary_search for ', N, 'elements, did', count, 'operations')
#     bin_search_counts.append(count)
    

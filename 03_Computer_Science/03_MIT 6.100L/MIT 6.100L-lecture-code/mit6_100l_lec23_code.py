import math
import time

## ----------------------------------
## CONSTANT Theta(1)
## ----------------------------------

## Theta(1)
def add(x, y):
    return x+y

## Theta(1)
def convert_to_km(m):
    return m*1.609

## ----------------------------------
## LINEAR Theta(n)
## Specify what n is in terms of input
## ----------------------------------

# constant in x: Theta(1)
# linear in y: Theta(y)
def mul(x, y):
    tot = 0
    for i in range(y):
        tot += x
    return tot

## Theta(len(s))
def add_digits(s):
    val = 0
    for c in s:
        val += int(c)
    return val

## Theta(a)
def fact_iter(a):
    prod = 1
    for i in range(1, a+1):
        prod *= i
    return prod

## Theta(x)
def fact_recur(x):
    """ assume x >= 0 """
    if x <= 1: 
        return 1
    else: 
        return x*fact_recur(x - 1)

## Theta(n_months)
def compound(invest, interest, n_months):
    total=0
    for i in range(n_months):
       total = total * interest + invest
    return total

## Theta(n)
def fib_iter(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        fib_i = 0
        fib_ii = 1
        for i in range(n-1):
            tmp = fib_i
            fib_i = fib_ii
            fib_ii = tmp + fib_ii
        return fib_ii 


## ----------------------------------
## POLYNOMIAL Theta(n**2)
## Specify what n is in terms of input
## ----------------------------------

## Theta(n**2)
def g(n):
   """ assume n >= 0 """
   x = 0
   for i in range(n):
      for j in range(n):
         x += 1
   return x

## Theta(len(L1)*len(L2))
## Theta(len(L1)**2)
def is_subset(L1, L2):
    for e1 in L1:
        matched = False
        for e2 in L2:
            if e1 == e2:
                matched = True
                break
        if not matched:
            return False
    return True

## Theta(len(L1)*len(L2))
def intersect(L1, L2):
    tmp = []
    for e1 in L1:
        for e2 in L2:
            if e1 == e2:
               tmp.append(e1)
    unique = []
    for e in tmp:
        if not(e in unique):
            unique.append(e)
    return unique

## Theta(len(L)**2)
def diameter(L):
    farthest_dist = 0
    for i in range(len(L)):
        for j in range(i+1, len(L)):
            p1 = L[i]
            p2 = L[j]
            dist = math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
            if dist > farthest_dist:
                farthest_dist = dist
    return farthest_dist


## ----------------------------------
## EXPONENTIAL Theta(2**n)
## Specify what n is in terms of input
## ----------------------------------

## Theta(2**len(L))
def gen_subsets(L):
    if len(L) == 0:
        return [[]] 
    extra = L[-1:]
    smaller = gen_subsets(L[:-1])
    new = []
    for small in smaller:
        new.append(small+extra)
    return smaller+new

#print(gen_subsets([1,2,3]))

## Theta(2**x)
def fib_recur(x):
    """ assumes x an int >= 0 """
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return fib_recur(x-1) + fib_recur(x-2)


## ----------------------------------
## LOGARITHMIC Theta(log(n))
## Specify what n is in terms of input
## ----------------------------------

## Theta(len(s)) but s is not the input!
## Theta(log(n))
def digit_sum(n):
    """ assumes n an int >= 0 """
    answer = 0
    s = str(n)
    for c in s:
        answer += int(c)
    return answer

    
## ----------------------------------
## SEARCHING ALGORITHMS
## ----------------------------------

####### searching on unsorted list
def linear_search(L, e):
    found = False
    for i in range(len(L)):
        if e == L[i]:
            found = True
    return found

####### searching on sorted list
def search(L, e):
    for i in range(len(L)):
        if L[i] == e:
            return True
        if L[i] > e:
            return False
    return False

#################################
## Bisection search with copying list
#################################    
def bisect_search1(L, e):
    """
    Assumes L is a list with elements in ascending order.
    Returns True if e is in L and False otherwise
    """
    if L == []:
        return False
    elif len(L) == 1:
        return L[0] == e
    else:
        half = len(L)//2
        if L[half] > e:
            return bisect_search1(L[:half], e)
        else:
            return bisect_search1(L[half:], e)

#################################
## Bisection search with indices
#################################    
def bisect_search2(L, e):
    """
    Assumes L is a list with elements in ascending order.
    Returns True if e is in L and False otherwise
    """
    def bisect_search_helper(L, e, low, high):
        if high == low:
            return L[low] == e
        mid = (low + high)//2
        if L[mid] == e:
            return True
        elif L[mid] > e:
            if low == mid: #nothing left to search
                return False
            else:
                return bisect_search_helper(L, e, low, mid - 1)
        else:
            return bisect_search_helper(L, e, mid + 1, high)
    if len(L) == 0:
        return False
    else:
        return bisect_search_helper(L, e, 0, len(L) - 1)

# print('-------')
# for i in [1000,10000,100000,1000000,10000000,100000000]:
#     # list comprehension 
#     # -> create a list where each element is x where x is each value in range(i)
#     L = [x for x in range(i)] 
#     e = -1
#     t0 = time.perf_counter()
#     print(linear_search(L, e), "with linear_search", i, "took", round(time.perf_counter() - t0, 6))
# print('-------')
# for i in [1000,10000,100000,1000000,10000000,100000000]:
#     # list comprehension 
#     # -> create a list where each element is x where x is each value in range(i)
#     L = [x for x in range(i)] 
#     e = -1
#     t0 = time.perf_counter()
#     print(bisect_search1(L, e), "with bisect_search1", i, "took", round(time.perf_counter() - t0, 6))
# print('-------')
# for i in [1000,10000,100000,1000000,10000000,100000000]:
#     # list comprehension 
#     # -> create a list where each element is x where x is each value in range(i)
#     L = [x for x in range(i)]
#     e = -1
#     t0 = time.perf_counter()
#     print(bisect_search2(L, e), "with bisect_search2", i, "took", round(time.perf_counter() - t0, 6))

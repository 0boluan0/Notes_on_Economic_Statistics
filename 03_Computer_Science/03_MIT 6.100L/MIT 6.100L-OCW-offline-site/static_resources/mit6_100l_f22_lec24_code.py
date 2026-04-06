import random
import time

############################
## Bogo/Random/Monkey Sort Example
############################
def is_sorted(L):
    i = 0
    j = len(L)
    while i + 1 < j:
        if L[i] > L[i + 1]:
            return False
        i += 1
    return True
 
def bogo_sort(L):
    count = 0
    while not is_sorted(L):
        random.shuffle(L)
        count += 1
    return count
 
# print("--- BOGO SORT ---")
# # L = []
# # for i in range(0, 9):
# #     L.append(random.randint(0, 100))
# L = [8, 4, 1, 6, 5, 11, 2, 0]
# print('L:       ', L)
# t0 = time.time()
# count = bogo_sort(L)
# print('Sorted L:', L)
# print(count, "shuffles and sorting took: ", time.time() - t0, "s")

############################
## Bubble Sort Example
############################
def bubble_sort(L, detail = False):
    did_swap = True
    while did_swap:
        did_swap = False
        for j in range(1, len(L)):
            if L[j-1] > L[j]:
                did_swap = True
                L[j],L[j-1] = L[j-1],L[j]
            if detail == True:
                print(L)
        print()

# print("--- BUBBLE SORT ---")
# L = [8, 4, 1, 6, 5, 11, 2, 0]
# print('L:       ', L)
# bubble_sort(L, True)
# print('Sorted L:', L)
  

############################
## Selection Sort Example
############################
def selection_sort(L, detail = False):
    for i in range(len(L)):
        for j in range(i, len(L)):
            if L[j] < L[i]:
                L[i], L[j] = L[j], L[i]
            if detail == True:
                print(L)
        print()

# print("--- SELECTION SORT ---")
# L = [8, 4, 1, 6, 5, 11, 2, 0]
# print('L:       ', L)
# selection_sort(L, True)
# print('Sorted L:', L)


############################
## Variation on Selection Sort Example
## Don't swap every time
############################
def selection_sort_var(L, detail = False):
    for i in range(len(L)):
        smallest = L[i]
        smallestj = i
        for j in range(i, len(L)):
            if L[j] < L[smallestj]:
                smallest = L[j]
                smallestj = j
            if detail == True:
                print(L)
        L[i], L[smallestj] = L[smallestj], L[i]
        print()

# print("--- VARIATION SELECTION SORT ---")
# L = [8, 4, 1, 6, 5, 11, 2, 0]
# print('L:       ', L)
# selection_sort_var(L, True)
# print('Sorted L:', L)


############################
## Merge Sort Example
############################
def merge(left, right):
    result = []
    i,j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    # one list is empty
    while (i < len(left)):
        result.append(left[i])
        i += 1
    while (j < len(right)):
        result.append(right[j])
        j += 1
    return result
    
def merge_sort(L, detail = False):
    if len(L) < 2:
        return L[:]
    else:
        middle = len(L)//2
        # divide
        left = merge_sort(L[:middle], detail)
        right = merge_sort(L[middle:], detail)
        if detail == True:
            print("Merging", left, "and", right)
        # conquer
        return merge(left, right)
    
# print("--- MERGE SORT ---")
# L = [8, 4, 1, 6, 5, 11, 2, 0]
# print('L:       ', L)
# print(merge_sort(L, True))
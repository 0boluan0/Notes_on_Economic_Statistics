
## remove from a list
# L = [2,1,3,6,3,7,0]
# L.remove(2)
# L.remove(3)
# del(L[1])
# print(L.pop())


#############
## Removing elements
#############

# L = [2,1,3,6,3,7,0] 
# L.pop(5)
# print(L)
# L.remove(3)
# print(L)
# L.pop()
# print(L)


############ YOU TRY IT ###############
# This one is similar to remove_elem from lec10 except that remove_elem 
# returns a new list and this one mutates the parameter L (and returns None)
def remove_all(L, e):
    """ 
    L is a list
    Mutates L to remove all elements in L that are equal to e
    Returns None.
    """
    # your code here
    

# Lin = [1,2,2,2]
# remove_all(Lin, 2)
# print(Lin)    # prints [1]

# Lin = [1,2,2,2]
# remove_all(Lin, 1)
# print(Lin)    # prints [2, 2, 2]

# Lin = [1,2,2,2]
# remove_all(Lin, 0)
# print(Lin)    # prints [1, 2, 2, 2]

#######################################

def remove_all(L, e):
    """ 
    L is a list
    Mutates L to remove all elements in L that are equal to e
    Returns None.
    """
    while e in L:
        L.remove(e)

# Lin = [1,2,2,2]
# remove_all(Lin, 2)
# print(Lin)    # prints [1]


## this function does not do the right thing
def remove_all(L, e):
    """ 
    L is a list
    Mutates L to remove all elements in L that are equal to e
    Returns None.
    """
    for elem in L:
        if elem == e:
            L.remove(e)

# Lin = [1,2,2,2]
# remove_all(Lin, 2)
# print(Lin)    # INCORRECTLY prints [1,2]



#############
## TRICKY EXAMPLE 4: removing element as you are mutating a list
#############
## this is an incorrect way to do it
def remove_dups(L1, L2):
    for e in L1:
        if e in L2:
            L1.remove(e)

# L1 = [10, 20, 30, 40]
# L2 = [10, 20, 50, 60]
# remove_dups(L1, L2)
# print(L1)

## this is an incorrect way to do it
def remove_dups(L1, L2):
    L1_copy = L1 # not actually a copy, just an alias!!
    for e in L1:
        if e in L2:
            L1.remove(e)

# L1 = [1, 2, 3, 4]
# L2 = [1, 2, 5, 6]
# remove_dups(L1, L2)
# print(L1)

## this is the CORRECT way to do it
def remove_dups(L1, L2):
    L1_copy = L1[:] # actually a copy aka clone
    for e in L1_copy:
        if e in L2:
            L1.remove(e)

# L1 = [1, 2, 3, 4]
# L2 = [1, 2, 5, 6]
# remove_dups(L1, L2)
# print(L1)




############################
############################
## Control copying, alises
# old_list = [[1,2],[3,4],[5,'foo']]
# new_list = old_list

# new_list[2][1] = 6
# print("New list:", new_list)
# print("Old list:", old_list)

## Control copying, shallow copy
# import copy
# old_list = [[1,2],[3,4],[5,6]]
# new_list = copy.copy(old_list)

# old_list.append([7,8])
# old_list[1][1] = 9
# print("New list:", new_list)
# print("Old list:", old_list)

## Control copying, deep copy
# import copy
# old_list = [[1,2],[3,4],[5,6]]
# new_list = copy.deepcopy(old_list)

# old_list.append([7,8])
# old_list[1][1] = 9
# print("New list:", new_list)
# print("Old list:", old_list)



## EXAMPLE: aliasing
# a = 1
# b = a
# print(a)
# print(b)

# warm = ['red', 'yellow', 'orange']
# hot = warm
# hot.append('pink')
# print(hot)
# print(warm)

## EXAMPLE: cloning
# cool = ['blue', 'green', 'grey']
# chill = cool[:]
# chill.append('black')
# print(chill)
# print(cool)

## EXAMPLE: sorting with/without mutation
# warm = ['red', 'yellow', 'orange']
# sortedwarm = warm.sort()
# print(warm)
# print(sortedwarm)

# cool = ['grey', 'green', 'blue']
# sortedcool = sorted(cool)
# print(cool)
# print(sortedcool)

## EXAMPLE: lists of lists of lists...
# warm = ['yellow', 'orange']
# hot = ['red']
# brightcolors = [warm]
# brightcolors.append(hot)
# print(brightcolors)
# hot.append('pink')
# print(hot)
# print(brightcolors)


############ YOU TRY IT AT HOME ###################
# Step through the code below without running it
# Write down what values each variable has
# Draw the memory diagram to help you keep track of aliases and clones

# cool = ['blue', 'green']
# warm = ['red', 'yellow', 'orange']
# print(cool)
# print(warm)

# colors1 = [cool]
# print(colors1)
# colors1.append(warm)
# print('colors1 = ', colors1)

# colors2 = [['blue', 'green'],
#           ['red', 'yellow', 'orange']]
# print('colors2 =', colors2)

# warm.remove('red') 
# print('colors1 = ', colors1)
# print('colors2 =', colors2)

# for e in colors1:
#     print('e =', e)

# for e in colors1:
#     if type(e) == list:
#         for e1 in e:
#             print(e1)
#     else:
#         print(e)

# flat = cool + warm
# print('flat =', flat)

# print(flat.sort())
# print('flat =', flat)

# new_flat = sorted(flat, reverse = True)
# print('flat =', flat)
# print('new_flat =', new_flat)

# cool[1] = 'black'
# print(cool)
# print(colors1)

###############################




############################################
################### AT HOME ######################
############################################
def repeat(L, n):
    """ L is a list of ints
        n is a positive int
    Mutates L to contain whatever elements L has right now repeated n times. """
    # your code here 
    
# Lin = [1,2,3]
# repeat(Lin, 3)
# print(Lin)    # prints [1, 2, 3, 1, 2, 3, 1, 2, 3]

# Think about why the following solution does not work!
def repeat(L, n):
    """ L is a list of ints
        n is a positive int
    Mutates L to contain whatever elements L has right now repeated n times. """
    # your code here 
    Lnew = []
    for i in range(n):
        for e in L:
            Lnew.append(e)
    Lin = Lnew  # hint, even thought we reuse the name Lin here, we make it point to a NEW object!
    
# Lin = [1,2,3]
# repeat(Lin, 3)
# print(Lin)   # prints [1, 2, 3] which is wrong!


#######################################
########## ANSWERS TO YOU TRY IT ###############
###########################################

def remove_all(L, e):
    """ 
    L is a list
    Mutates L to remove all elements in L that are equal to e
    """ 
    Lnew = L[:]
    L.clear()
    for elem in Lnew:
        if elem != e:
            L.append(elem)

# L = [1,2,2,2]
# remove_all(L, 1)
# print(L)    # prints [2, 2, 2]

# L = [1,2,2,2]
# remove_all(L, 2)
# print(L)    # prints [1]

# L = [1,2,2,2]
# remove_all(L, 0)
# print(L)    # prints [1, 2, 2, 2]



#######################################
########## ANSWERS TO AT HOME ###############
###########################################
def repeat(L, n):
    """ L is a list of ints
        n is a positive int
    Mutates L to contain whatever elements L has right now repeated n times. """
    # your code here 
    rep = len(L)
    for i in range(n-1):
        for j in range(rep):
            L.append(L[j])
    
# Lin = [1,2,3]
# repeat(Lin, 3)
# print(Lin)    # prints [1, 2, 3, 1, 2, 3, 1, 2, 3]
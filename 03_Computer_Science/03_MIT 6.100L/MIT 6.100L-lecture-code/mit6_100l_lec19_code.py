import random

#################################
## Animal abstract data type 
#################################
class Animal(object):
    def __init__(self, age):
        self.age = age
        self.name = None
    def __str__(self):
        return "animal:"+str(self.name)+":"+str(self.age)
    def get_age(self):
        return self.age
    def get_name(self):
        return self.name
    def set_age(self, newage):
        self.age = newage
    def set_name(self, newname=""):
        self.name = newname

# #default parameters with methods        
a = Animal(4)
# print(a)
b = Animal(6)
# print(b)
# print(a.age)
# print(a.get_age())

# a.set_name("fluffy")
# print(a.name)
# print(a.get_name())
# print(a)
# a.set_name()
# print(a)

# #Accessing data attributes and adding new attributes (just for one instance)
# #These are bad to do, use getter and setter methods instead!
# a = Animal(4)
# a.name = "furball"
# a.age = "twelve"
# a.size = "tiny"
# print(a.get_age())


### EXAMPLE: using Animal objects in code
def animal_dict(L):
    """ L is a list
    Returns a dict, d, mappping an int to an Animal object. 
    A key in d is all non-negative ints, n, in L. A value 
    corresponding to a key is an Animal object with n as its age. """
    d = {}
    for n in L:
        if type(n) == int and n >= 0:
            d[n] = Animal(n)
    return d

L = [2,5,'a',-5,0]
animals = animal_dict(L)

# print(animals)
# above prints {2: <__main__.Animal object at 0x00000199AFF350A0>, 
#               5: <__main__.Animal object at 0x00000199AFF35A30>, 
#               0: <__main__.Animal object at 0x00000199AFF35D00>}

# for n,a in animals.items():   
#     print(f'key {n} with val {a}')
# loop above prints animal:None:2 
#                   animal:None:5 
#                   animal:None:0

###################### YOU TRY IT ####################
# Write a function that meets this spec.
def make_animals(L1, L2):
    """ L1 is a list if ints and L2 is a list of str
        L1 and L2 have the same length
    Creates a list of Animals the same length as L1 and L2.
    An animal object at index i has the age and name
    corresponding to the same index in L1 and L2, respectively. """
    # your code here


L1 = [2,5,1]
L2 = ["blobfish", "crazyant", "parafox"]
# animals = make_animals(L1, L2)
# print(animals)     # note this prints a list of animal objects
# for i in animals:  # this prints the indivdual animals
#     print(i)

#######################################################


#################################
## Inheritance example 
#################################
class Cat(Animal):
    def speak(self):
        print("meow")
    def __str__(self):
        return "cat:"+str(self.name)+":"+str(self.age)
    
#print("\n---- cat tests ----")
c = Cat(5)
# c.set_name("fluffy")
# print(c)
# c.speak()
# print(c.get_age())
# a.speak() # error because there is no speak method for Animal class


################# YOU TRY IT #####################
class Rabbit(Animal):
    """ A subclass of Animal """
    def speak(self):
        """ prints the string meep to the console """
        # your code here

    def __str__(self):
        """ Repr as "rabbit", a colon, self's name, a colon, self's age """
        # your code here
   
r = Rabbit(5)
# print(r)
# r.speak()
# r.set_name('fluffy')
# print(r)
    
    
##################################
### Inheritance example
##################################
class Person(Animal):
    def __init__(self, name, age):
        Animal.__init__(self, age)
        self.set_name(name)
        self.friends = []
    def get_friends(self):
        return self.friends.copy()
    def speak(self):
        print("hello")
    def add_friend(self, fname):
        if fname not in self.friends:
            self.friends.append(fname)
    def age_diff(self, other):
        diff = self.age - other.age
        print(abs(diff), "year difference")
    def __str__(self):
        return "person:"+str(self.name)+":"+str(self.age)

#print("\n---- person tests ----")
p1 = Person("jack", 30)
p2 = Person("jill", 25)
# print(p1.get_name())
# print(p1.get_age())
# print(p2.get_name())
# print(p2.get_age())
# print(p1)
# p1.speak()
# p1.age_diff(p2)
# p1.add_friend('ana')
# p1.add_friend('bob')
# p1.add_friend('bob')
# print(p1.get_friends())

######################## YOU TRY IT #####################
# Write the function according to this spec
def make_pets(d):
    """ d is a dict mapping a Person obj to a Cat obj
    Prints, on each line, the name of a person, 
    a colon, and the name of that person's cat """
    # your code here


p1 = Person("ana", 86)
p2 = Person("james", 7)
c1 = Cat(1)
c1.set_name("furball")
c2 = Cat(1)
c2.set_name("fluffsphere")

# d = {p1:c1, p2:c2}
# make_pets(d)  # prints ana:furball
       #        james:fluffsphere

##########################################################

##################################
### Inheritance example
##################################
class Student(Person):
    def __init__(self, name, age, major=None):
        Person.__init__(self, name, age)
        self.major = major
    def __str__(self):
        return "student:"+str(self.name)+":"+str(self.age)+":"+str(self.major)
    def change_major(self, major):
        self.major = major
    def speak(self):
        r = random.random()
        if r < 0.25:
            print("--> i have homework")
        elif 0.25 <= r < 0.5:
            print("--> i need sleep")
        elif 0.5 <= r < 0.75:
            print("--> i should eat")
        else:
            print("--> i'm zooming")

# print("\n---- student tests ----")
s1 = Student('alice', 20, "CS")
s2 = Student('beth', 18)
# print(s1)
# print(s2)
# print(s1.get_name(),"says:")
# s1.speak()
# print(s2.get_name(),"says:")
# s2.speak()


##################################
### Use of class variables  
##################################
class Rabbit(Animal):
    # a class variable, tag, shared across all instances
    tag = 1
    def __init__(self, age, parent1=None, parent2=None):
        Animal.__init__(self, age)
        self.p1 = parent1
        self.p2 = parent2
        self.rid = Rabbit.tag
        Rabbit.tag += 1
    def get_rid(self):
        # zfill used to add leading zeroes 
        # 00001 instead of 1 or 00526 instead of 526
        return str(self.rid).zfill(5)
    def get_parent1(self):
        return self.p1
    def get_parent2(self):
        return self.p2
    def __add__(self, oth):
        # returning object of same type as this class
        return Rabbit(0, self, oth)
    def __eq__(self, oth):
        # compare the ids of self and other's parents
        # don't care about the order of the parents
        parents_same = (self.p1.rid == oth.p1.rid and self.p2.rid == oth.p2.rid)
        parents_opp = (self.p2.rid == oth.p1.rid and self.p1.rid == oth.p2.rid)
        return parents_same or parents_opp
    def __str__(self):
        return "rabbit:"+ self.get_rid()

# # print("\n---- rabbit tests ----")
# # print("---- testing creating rabbits ----")
r1 = Rabbit(3)
r2 = Rabbit(4)
r3 = Rabbit(5)
# print("r1:", r1)
# print("r2:", r2)
# print("r3:", r3)
# print("r1 parent1:", r1.get_parent1())
# print("r1 parent2:", r1.get_parent2())

# # print("---- testing rabbit addition ----")
r4 = r1+r2   # r1.__add__(r2)
# print("r1:", r1)
# print("r2:", r2)
# print("r4:", r4)
# print("r4 parent1:", r4.get_parent1())
# print("r4 parent2:", r4.get_parent2())

# # print("---- testing rabbit equality ----")
r5 = r3+r4
r6 = r4+r3
# print("r3:", r3)
# print("r4:", r4)
# print("r5:", r5)
# print("r6:", r6)
# print("r5 parent1:", r5.get_parent1())
# print("r5 parent2:", r5.get_parent2())
# print("r6 parent1:", r6.get_parent1())
# print("r6 parent2:", r6.get_parent2())
# print("r5 and r6 have same parents?", r5 == r6)
# print("r4 and r6 have same parents?", r4 == r6)


##################################################
######### ANSWERS TO YOU TRY IT ###############
##################################################

# Q1. Write a function that meets this spec.
def make_animals(L1, L2):
    """ L1 is a list if ints and L2 is a list of str
        L1 and L2 have the same length
    Creates a list of Animals the same length as L1 and L2.
    An animal object at index i has the age and name
    corresponding to the same index in L1 and L2, respectively. """
    pass
    L = []
    for i in range(len(L1)):  # i will be 0 then 1 then 2
        a = Animal(L1[i]) # L1[i] is the age
        a.set_name(L2[i]) # L2[i] is the name
        L.append(a)
    return L        

# L1 = [2,5,1]
# L2 = ["blobfish", "crazyant", "parafox"]
# animals = make_animals(L1, L2)
# print(animals)     # note this prints a list of animal objects
# for i in animals:  # this prints the indivdual animals
#     print(i)

# Q2. 
class Rabbit(Animal):
    """ A subclass of Animal """
    def speak(self):
        """ prints the string meep to the console """
        print('meep')
    def __str__(self):
        """ Repr as "rabbit", a colon, its name, a colon, its age """
        return "rabbit:"+str(self.get_name())+":"+str(self.get_age())
   
# c = Rabbit(5)
# print(c)
# c.speak()
# c.set_name('fluffy')
# print(c)


# Q3. Write the function according to this spec
def make_pets(d):
    """ d is a dict mapping a Person obj to a Cat obj
    Prints the name of each person, a colon, and the 
    name of that person's cat """
    pass
    for k,v in d.items():
        # k is Person
        # v is Cat
        pname = k.get_name()
        cname = v.get_name()
        print(pname+":"+cname)


# p1 = Person("ana", 35)
# p2 = Person("james", 6)
# c1 = Cat(1)
# c1.set_name("furball")
# c2 = Cat(1)
# c2.set_name("fluffsphere")

# d = {p1:c1, p2:c2}
# make_pets(d)  # prints ana:furball
#       #        james:fluffsphere


##################################################
######### AT HOME ###############
##################################################
# Write the class Employee as a subclass of Person
class Employee(Person):
    """ An Employee contains an extra data attribute, salary as an int """
    def __init__(self, name, age):
        """ initializes self as a Person with a salary data attribute, initially 0 """
        pass
    def get_salary(self):
        """ returns self's salary """
        pass
    def set_salary(self, s):
        """ s is an int
        Sets self's salary data attribute to s """
        pass
    def salary_change(self, n):
        """ n is an int (positive or negative)
        Adds n to self's salary. If the result is negative, sets 
        self's salary to 0. Otherwise sets self's salary to the new value. """
        pass
    def has_friends(self):
        """ Returns True if self's friend list is empty and False otherwise """
        pass    
    def past_salaries_list(self):
        """ Keeps track of all salaries self has had in the order they've changed. 
        i.e. whenever the salary changes, keep track of it.
        Hint: you may need to add an additional data attribute to Employee.
        Returns a copy of the list of all salaries self has had, in order. """
        pass
    def past_salary_freq(self):
        """ Returns a dictionary where the key is a salary number and the 
        value is how many times self's salary has changed to that number. """
        pass

# # For example:
# e = Employee("ana", 35)
# print(e.get_salary())   # prints 0
# e.set_salary(1000)
# print(e.get_salary())   # prints 1000
# e.salary_change(2000)
# print(e.get_salary())   # prints 3000
# e.salary_change(-50000)
# print(e.get_salary())   # prints 0
# print(e.has_friends())  # prints False
# e.add_friend("bob")
# print(e.has_friends())  # prints True
# print(e.past_salaries_list())  # prints [0, 1000, 3000, 0]
# print(e.past_salary_freq())  # prints {0: 2, 1000: 1, 3000: 1}


# Write a function that meets this specification
def counts(L):
    """ L is a list of Employee and Person objects 
    Returns a tuple of a count of:
      * how many Person objects are in L
      * how many Employee objects are in L 
      * the number of unique names among Employee and Person objects """
    pass

# For example:
# make employees and people
# L = []
# L.append(Person('ana',8))
# L.append(Person('bob',25))
# L.append(Employee('ana',35))
# L.append(Employee('cara',18))
# L.append(Employee('dan',53))
# # call function
# print(counts(L))    # prints (2,3,4)


##################################################
######### ANSWERS TO AT HOME #######
################################################
class Employee(Person):
    """ An Employee contains an extra data attribute, salary as an int """
    def __init__(self, name, age):
        """ initializes self as a Person with a salary data attribute, initially 0 """
        Person.__init__(self,name, age)
        self.salary = 0
        self.list_salaries = [0]
    def get_salary(self):
        """ returns self's salary """
        return self.salary
    def set_salary(self, s):
        """ s is an int
        Sets self's salary data attribute to s """
        self.salary = s
        self.list_salaries.append(s)
    def salary_change(self, n):
        """ n is an int (positive or negative)
        Adds n to self's salary. If the result is negative, sets 
        self's salary to 0. Otherwise sets self's salary to the new value. """
        a = self.salary + n
        if a < 0:
            self.salary = 0
        else:
            self.salary = a
        self.list_salaries.append(self.salary)
    def has_friends(self):
        """ Returns True if self's friend list is empty and False otherwise """
        return len(self.friends) != 0
    def past_salaries_list(self):
        """ Keeps track of all salaries self has had in the order they've changed. 
        i.e. whenever the salary changes, keep track of it.
        Hint: you may need to add an additional data attribute to Employee.
        Returns a copy of the list of all salaries self has had, in order. """
        return self.list_salaries.copy()
    def past_salary_freq(self):
        """ Keeps track of all salaries self has had in the order they've changed. 
        i.e. whenever the salary changes, keep track of it.
        Hint: you may need to add an additional data attribute to Employee.
        Returns a dictionary where the key is a salary number and the value 
        is how many times self's salary has changed to that value. """
        d = {}
        for i in self.list_salaries:
            if i in d:
                d[i] += 1
            else:
                d[i] = 1
        return d
    
# # For example:
# e = Employee("ana", 35)
# print(e.get_salary())   # prints 0
# e.set_salary(1000)
# print(e.get_salary())   # prints 1000
# e.salary_change(2000)
# print(e.get_salary())   # prints 3000
# e.salary_change(-50000)
# print(e.get_salary())   # prints 0
# print(e.has_friends())  # prints False
# e.add_friend("bob")
# print(e.has_friends())  # prints True
# print(e.past_salaries_list())  # prints [0, 1000, 3000, 0]
# print(e.past_salary_freq())  # prints {0: 2, 1000: 1, 3000: 1}


# def counts(L):
#     """ L is a list of Employee and Person objects 
#     Returns a tuple of a count of:
#       * how many Person objects are in L
#       * how many Employee objects are in L 
#       * the number of unique names among Employee and Person objects """
#     counte, countp, countn = 0,0,0
#     names = []
#     for i in L:
#         if type(i) == Person:
#             countp += 1
#         elif type(i) == Employee:
#             counte += 1
#         if i.get_name() not in names:
#             names.append(i.get_name())
#             countn += 1
#     return (countp, counte, countn)

# For example:
# make employees and people
# L = []
# L.append(Person('ana',8))
# L.append(Person('bob',25))
# L.append(Employee('ana',35))
# L.append(Employee('cara',18))
# L.append(Employee('dan',53))
# # call function
# print(counts(L))    # prints (2,3,4)
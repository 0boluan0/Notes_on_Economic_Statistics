## EXAMPLE: simple Coordinate class
class Coordinate(object):
    """ A coordinate made up of an x and y value """
    def __init__(self, x, y):
        """ Sets the x and y values """
        self.x = x
        self.y = y
    def distance(self, other):
        """ Returns the euclidean distance between two Coordinate objects """
        x_diff_sq = (self.x-other.x)**2
        y_diff_sq = (self.y-other.y)**2
        return (x_diff_sq + y_diff_sq)**0.5
    def to_origin(self):
        """ always sets self.x and self.y to 0,0 """
        self.x = 0
        self.y = 0
    def __str__(self):
        """ Returns a string representation of self """
        return "<" + str(self.x) + "," + str(self.y) + ">"


# #Print a coordinate object's data attributes
c = Coordinate(3,4)
origin = Coordinate(0,0)
# print(f"c's x is {c.x} and origin's x is {origin.x}")

# #These are equivalent calls
#print(c.distance(origin))
#print(Coordinate.distance(c, origin))

# #Calling a new method
# c.to_origin()
# print(c.x, c.y)

# #Printing a coordinate object
# print(c)
# print(origin)
#c+origin


############## YOU TRY IT #######################
# Add code to the init method to check that 
# * the type of center is a Coordinate obj and 
# * the type of radius is an int. 
# If either are not these types, raise a ValueError.
class Circle(object):
    def __init__(self, center, radius):

        self.center = center
        self.radius = radius
        

# center = Coordinate(2, 2)
# my_circle = Circle(center, 2)   # no error

# my_circle = Circle(2, 2)    # raises ValueError
# my_circle = Circle(center, 'two')  # raises ValueError

##################################################

## EXAMPLE: use Coordinate objects to build Circle objects
class Circle(object):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def is_inside(self, point):
        """ Returns True if point is inside self and False otherwise """
        return point.distance(self.center) < self.radius
    
center = Coordinate(2, 2)
my_circle = Circle(center, 2)

# p = Coordinate(1,1)
# print(my_circle.is_inside(p))

# p = Coordinate(10,10)
# print(my_circle.is_inside(p))



## EXAMPLE: simple class to represent fractions
class SimpleFraction(object):
    """ A number represented as a fraction """
    def __init__(self, num, denom):
        """ num and denom are integers """
        self.num = num
        self.denom = denom
    def times(self, other):
        """ Returns a float representing the addition """
        top = self.num*other.num
        bottom = self.denom*other.denom
        return top/bottom
    def divide(self, other):
        """ Returns a float representing the subtraction """
        top = self.num*other.denom
        bottom = self.denom*other.num
        return top/bottom
    def plus(self, other):
        """ Returns a float representing the addition """
        top = self.num*other.denom + self.denom*other.num
        bottom = self.denom*other.denom
        return top/bottom
    def minus(self, other):
        """ Returns a float representing the subtraction """
        top = self.num*other.denom - self.denom*other.num
        bottom = self.denom*other.denom
        return top/bottom


f1 = SimpleFraction(3, 4)
f2 = SimpleFraction(1, 4)
# print(f1.num)
# print(f1.denom)
# print(f2.num)
# print(f2.denom)
# print(f1.times(f2))

# print(f1.plus(f2))
# print(f1.divide(f2))
# print(f1.minus(f2))

# print(f1)
# print(f1.times(f2))
# print(f1 * f2)  # given an error


########### YOU TRY IT ##################
# Implement the missing get_inverse and invert methods below
class SimpleFraction(object):
    """ A number represented as a fraction """
    def __init__(self, num, denom):
        """ num and denom are integers """
        self.num = num
        self.denom = denom
    def get_inverse(self):
        """ Returns a float representing 1/self """
        # your code here
        
    def invert(self):
        """ Sets self's numerator to its denominator and vice versa.
            Returns None. """
        # your code here
        
        
# f1 = SimpleFraction(3,4)
# print(f1.num, f1.denom)   # prints 3 4 
# print(f1.get_inverse())   # prints 1.33333333 (note this one returns value)
# f1.invert()               # acts on data attributes internally, no return
# print(f1.num, f1.denom)   # prints 4 3 


#########################################


## EXAMPLE: simple class to represent fractions
## Added functionality by implementing +, -, *, / operators
class Fraction(object):
    """ A number represented as a fraction """
    def __init__(self, num, denom):
        """ num and denom are integers """
        self.num = num
        self.denom = denom
    def __str__(self):
        """ Returns a string representation of self """
        return str(self.num) + "/" + str(self.denom)
    def __mul__(self, other):
        """ Returns a new fraction representing the addition """
        top = self.num*other.num
        bottom = self.denom*other.denom
        return Fraction(top, bottom)
    def __add__(self, other):
        """ Returns a new fraction representing the addition """
        top = self.num*other.denom + self.denom*other.num
        bottom = self.denom*other.denom
        return Fraction(top, bottom)
    def __sub__(self, other):
        """ Returns a new fraction representing the subtraction """
        top = self.num*other.denom - self.denom*other.num
        bottom = self.denom*other.denom
        return Fraction(top, bottom)
    def __truediv__(self, other):
        """ Returns a new fraction representing the subtraction """
        top = self.num*other.denom
        bottom = self.denom*other.num
        return Fraction(top, bottom)
    def __float__(self):
        """ Returns a float value of the fraction """
        return self.num/self.denom
    def reduce(self):
        """ Returns a new fraction the reduced version of self 
            using the greatest common divisor """
        def gcd(n, d):
            while d != 0:
                (d, n) = (n%d, d)
            return n
        if self.denom == 0:
            return None
        elif self.denom == 1:
            return self.num
        else:
            greatest_common_divisor = gcd(self.num,self.denom)
            top = int(self.num/greatest_common_divisor)
            bottom = int(self.denom/greatest_common_divisor)
            return Fraction(top, bottom)
    def invert(self):
        """ Returns a new fraction representing 1/self """
        return Fraction(self.denom, self.num)

# # Using shorthand operations on fractions        
# a = Fraction(1,4)
# b = Fraction(3,4)
# print(a)
# c = a * b # c is a Fraction object
# print(c)

# # The next 3 lines are equivalent
# print(a * b)                  ##1 (shorthand) same as #2, #3
# print(a.__mul__(b))           ##2 (method call) same as #1, #3
# print(Fraction.__mul__(a, b)) ##3 (explicit class call) same as #1, #2
# ########

# # The next 3 lines are equivalent
# print(float(c))              ##1 (shorthand) same as #2, #3
# print(c.__float__())         ##2 (method call) same as #1, #3
# print(Fraction.__float__(c)) ##3 (explicit class call) same as #1, #2

# # Reducing fractions
# a = Fraction(1,4)
# b = Fraction(2,3)
# c = a * b 
# print(c)
# print(c.reduce())

# # Can't multiply int and Fraction
# a = Fraction(4,1)
# b = Fraction(3,9)
# ar = a.reduce()
# br = b.reduce()
# print(ar, type(ar))
# print(br, type(br))
# # c = ar * br   # gives an error bc it's multiplying an int with a Fraction


############## YOU TRY IT #####################
# Modify the str method to represent the Fraction as just the 
# numerator, when the denominator is 1. Otherwise its representation 
# is the numerator then a / then the denominator, as before
class Fraction(object):
    """ A number represented as a fraction """
    def __init__(self, num, denom):
        """ num and denom are integers """
        self.num = num
        self.denom = denom
    def __str__(self):
        """ Returns a string representation of self """
        # modify this
        return str(self.num) + "/" + str(self.denom)

 
a = Fraction(1,4)
b = Fraction(3,1)
# print(a)     # prints 1/4
# print(b)     # prints 3

#######################################################

################ YOU TRY IT ############################
# Modify the code to return a Fraction object when denominator is 1
class Fraction(object):
    def __init__(self, num, denom):
        """ num and denom are integers """
        self.num = num
        self.denom = denom
    def reduce(self):
        def gcd(n, d):
            while d != 0:
                (d, n) = (n%d, d)
            return n
        if self.denom == 0:
            return None
        elif self.denom == 1:
            # modify this
            return self.num
        else:
            greatest_common_divisor = gcd(self.num, self.denom)
            top = int(self.num/greatest_common_divisor)
            bottom = int(self.denom/greatest_common_divisor)
            return Fraction(top, bottom)
    def __str__(self):
        """ Returns a string representation of self """
        # Note this is not the version with the numerator 
        # only when the denomiator is 1
        return str(self.num) + "/" + str(self.denom)
    
f1 = Fraction(5,1)
f1r = f1.reduce()
# print(f1r)          # prints 5/1 not 5
# print(type(f1r))    # prints <class '__main__.Fraction'>

####################################################



###########################################################
############### ANSWERS TO YOU TRY IT ####################
###########################################################
# Q1. Add code to the init method to check that 
# * the type of center is a Coordinate obj and 
# * the type of radius is an int. 
# If either are not these types, raise a ValueError.
class Circle(object):
    def __init__(self, center, radius):
        if type(center) == Coordinate and type(radius) == int:
            self.center = center
            self.radius = radius
        else:
            raise ValueError

# center = Coordinate(2, 2)
# my_circle = Circle(center, 2)   # no error

# my_circle = Circle(2, 2)    # raises ValueError
# my_circle = Circle(center, 'two')  # raises ValueError


# Q2. Implement the missing get_inverse and invert methods below
class SimpleFraction(object):
    """ A number represented as a fraction """
    def __init__(self, num, denom):
        """ num and denom are integers """
        self.num = num
        self.denom = denom
    def get_inverse(self):
        """ Returns a float representing 1/self """
        return self.denom/self.num
    def invert(self):
        """ Sets self's numerator to its denominator and vice versa.
            Does not return anything. """
        (self.num, self.denom) = (self.denom, self.num)

# f1 = SimpleFraction(3,4)
# print(f1.get_inverse())   # prints 1.33333333 (note this one returns value)
# f1.invert()               # acts on data attributes internally, no return
# print(f1.num, f1.denom)   # prints 4 3 


# Q3. Modify the str method to print just the numerator when 
# the denominator is 1. Otherwise it prints the numerator 
# then a / then the denominator, as before. 
class Fraction(object):
    """ A number represented as a fraction """
    def __init__(self, num, denom):
        """ num and denom are integers """
        self.num = num
        self.denom = denom
    def __str__(self):
        """ Returns a string representation of self """
        # modify this
        if self.denom == 1:
            return str(self.num)
        return str(self.num) + "/" + str(self.denom)

# a = Fraction(1,4)
# b = Fraction(3,1)
# print(a)     # prints 1/4
# print(b)     # prints 3

# Q4. Modify the code to return a Fraction object when denominator is 1
class Fraction(object):
    def __init__(self, num, denom):
        """ num and denom are integers """
        self.num = num
        self.denom = denom
    def reduce(self):
        def gcd(n, d):
            while d != 0:
                (d, n) = (n%d, d)
            return n
        if self.denom == 0:
            return None
        elif self.denom == 1:
            # modify this
            return Fraction(self.num,1)
        else:
            greatest_common_divisor = gcd(self.num, self.denom)
            top = int(self.num/greatest_common_divisor)
            bottom = int(self.denom/greatest_common_divisor)
            return Fraction(top, bottom)
    def __str__(self):
        """ Returns a string representation of self """
        return str(self.num) + "/" + str(self.denom)
    
# f1 = Fraction(5,1)
# f1r = f1.reduce()
# print(f1r)          # prints 5/1 not 5
# print(type(f1r))    # prints <class '__main__.Fraction'>


###########################################################
############### AT HOME ####################
###########################################################
#Question 1.
# Add a method to the Circle class that allows you to print a Circle object
# (you decide how to best represent it!)

#Question 2.
# Implement a method in Fraction class such that the operator ** works
#print(a**b) # works after you define it on two Fraction objects


###########################################################
############# ANSWERS TO AT HOME ###################
###########################################################
# Question 1.
# class Circle(object):
#     def __init__(self, center, radius):
#         self.center = center
#         self.radius = radius
#     def is_inside(self, point):
#         return point.distance(self.center) < self.radius
#     # one way
#     def __str__(self):
#         return "circle: "+str(self.center)+", "+str(self.radius)
#     # alternate cooler way :)
#     # prints radius number of dashes to the left and right of the center
#     def __str__(self):
#         return "-"*self.radius+str(self.center)+"-"*self.radius
    
# center = Coordinate(2, 2)
# my_circle = Circle(center, 5)
# print(my_circle)


# Question 2.
# class Fraction(object):
#     def __init__(self, num, denom):
#         self.num = num
#         self.denom = denom
#     def __float__(self):
#         return self.num/self.denom
#     def __str__(self):
#         return str(self.num) + "/" + str(self.denom)
#     def __pow__(self, other):
#         return float(self)**float(other)
    
# f1 = Fraction(4,1)
# f2 = Fraction(1,2)
# print(f1**f2)    # prints 2.0

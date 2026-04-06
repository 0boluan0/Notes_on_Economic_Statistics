## EXAMPLE from lecture: simple Coordinate class
class Coordinate(object):
    """ A coordinate made up of an x and y value """
    def __init__(self, x, y):
        """ Sets the x and y values """
        self.x = x
        self.y = y
    def distance(self, other):
        """ Returns the euclidean distance between two points """
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
    
c = Coordinate(1,2)
print(c.x)
print(c.y)

list1=[3,2]
c1 = Coordinate(1,3)
c1.x

c.to_origin()
print(c.x, c.y) 
c.distance(c1)
print(c)

## EXAMPLE Person Class
class Person(object):
    def __init__(self, age):
        self.age = age
        self.todo = []
    
    def birthday(self):
        self.age += 1
    
    def add_todo(self, item):
        self.todo += [item]
    
    def mark_off(self):
        self.todo = self.todo[1:]
        
    def get_age(self):
        return self.age


# Uncomment the following to use the code above (and add print statements to 
# test methods)

# Create instance of Person class
person = Person(21)  

# Testing different Class methods
# print(person.get_age())  
# person.birthday()  
# print(person.age)  
# print(person.get_age()) 
# print(person.todo)
# person.add_todo("Todo 1")
# person.add_todo("Todo 2")
# print(person.todo)
# person.mark_off()
# print(person.todo)





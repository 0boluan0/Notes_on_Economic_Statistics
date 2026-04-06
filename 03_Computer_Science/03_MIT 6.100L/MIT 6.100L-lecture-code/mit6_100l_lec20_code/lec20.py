from dateutil import parser
from lec20_helpers import gpsDistance


# =============================================================================
# EXAMPLE: Simple workout class
# =============================================================================
class SimpleWorkout(object):
    """A simple class to keep track of workouts"""
    def __init__(self, start, end, calories):
        self.start = start
        self.end = end
        self.calories = calories
        self.icon = '😓'
        self.kind = 'Workout'
    def get_calories(self):
        return self.calories
    def get_start(self):
        return self.start
    def get_end(self):
        return self.end
    def set_calories(self, calories):
        self.calories = calories
    def set_start(self, start):
        self.start = start
    def set_end(self, end):
        self.end = end


# =============================================================================
# TEST: Inspect internal state of classes
# =============================================================================
# print(SimpleWorkout.__dict__.keys())   # dict_keys(['__module__', '__init__', 'get_calories', 'get_start', 'set_calories', 'set_start', '__dict__', '__weakref__', '__doc__'])
print()
# print(SimpleWorkout.__dict__.values())   

my_workout = SimpleWorkout('9/30/2021 1:35 PM', '9/30/2021 1:57 PM', 200)
# print(my_workout.__dict__.keys())  # dict_keys(['start', 'end', 'calories', 'icon', 'kind'])
print()
# print(my_workout.__dict__.values()) 

                     
# =============================================================================
# EXAMPLE: Workout class
# =============================================================================
class Workout(object):
    """A class to keep track of workouts"""

    # Class variable to compute calories burned from workout time
    cal_per_hr = 200
    
    def __init__(self, start, end, calories=None):
        """Creates a workout class;  start and end are strings representing
        the start and end time (e.g., "1/1/2021 1:23 PM");  
        calories is an optional float specifying the calories burned 
        in the workout"""
        # note use of dateutil.parser to convert strings to datetime objects
        self.start = parser.parse(start)  
        self.end = parser.parse(end)
        self.icon = '😓'
        self.kind = 'Workout'
        self.calories = calories

    def get_calories(self):
        """Return the total calories burned in the workout"""
        if (self.calories == None):
            # calc the calories based on the length of the workout and cal_per_hr
            return Workout.cal_per_hr * (self.end - self.start).total_seconds() / 3600.0
        else:
            return self.calories

    def get_duration(self):
        """Return the duration of the workout, as a datetime.interval object"""
        return self.end-self.start

    def get_start(self):
        """Return the start time of the workout, as a datetime.datetime object"""
        return self.start

    def get_end(self):
        """Return the end time of the workout, as a datetime.datetime object"""
        return self.end

    def set_calories(self, calories):
        """Set the calories of the workout to calories"""
        self.calories = calories

    def set_start(self, start):
        """Set the start time of the workout to the supplied start string"""
        self.start = parser.parse(start)

    def set_end(self, end):
        """Set the start time of the workout to the supplied start string"""
        self.end = parser.parse(end)

    def get_kind(self):
        """Return the kind of the workout as a string"""
        return self.kind

    def __eq__(self, other):
        """Returns true if this workout is equal to another workout, false o.w."""
        # the \ breaks up the line
        return type(self) == type(other) and \
                self.start == other.start and \
                self.end == other.end and \
                self.kind == other.kind and \
                self.get_calories() == other.get_calories()

    def __str__(self):
        """Return a text-based graphical depiction of the workout"""
        width = 16
        retstr =  f"|{'–'*width}|\n"
        retstr += f"|{' ' *width}|\n"
        retstr += f"| {self.icon}{' '*(width-3)}|\n"  #assume width of icon is 2 chars - len('🏃🏽‍♀️');  doesn't do what you'd epxect
        retstr += f"| {self.kind}{' '*(width-len(self.kind)-1)}|\n"
        retstr += f"|{' ' *width}|\n"
        duration_str = str(self.get_duration())
        retstr += f"| {duration_str}{' '*(width-len(duration_str)-1)}|\n"
        cal_str = f"{round(self.get_calories(),1)}"
        retstr += f"| {cal_str} Calories {' '*(width-len(cal_str)-11)}|\n"

        retstr += f"|{' ' *width}|\n"
        retstr +=  f"|{'_'*width}|\n"

        return retstr


# =============================================================================
# TEST: Try get_calories() with and without calories being set
#        (Information Hiding Example)
# =============================================================================
# # Create a workout with calories set
# # Does not calculate cal number based on time diff, uses parameter directly
workout1 = Workout('9/30/2021 1:35 PM','9/30/2021 1:57 PM',400)
# print(workout1.get_calories())

# # Create a workout without calories set
# # Default is 200 cal per hr, calculates cal number based on time diff (22 mins)
workout2 = Workout('9/30/2021 1:35 PM','9/30/2021 1:57 PM')
# print(workout2.get_calories())


# =============================================================================
# TEST: Using the datetime object
# =============================================================================
from dateutil import parser
start = '9/30/2021 1:35 PM'
end = '9/30/2021 1:45 PM'
# it can read dates and times in any format! 
# start = 'Sept 30 2021 1:35 PM'
# end = 'September, 30, 2021 1:45pm'

# # convert the strings to datetime objects
# start_date = parser.parse(start)
# end_date = parser.parse(end)
# print(f'type of start_date: {type(start_date)}')
# print(f'type of end_date: {type(end_date)}')

# # operations with datetime objects are very easy because 
# # the datetime library deals with all the pesky details, yay!
# # subtracting two datetime objects gives us a timedelta object
# print((end_date-start_date))  # is the repr of a datetime obj
# print((end_date-start_date).total_seconds())  # in seconds as a float


# =============================================================================
# TEST: Class variables 
# =============================================================================
# Get the cal_per_hr class variable (note Workout. not instance_name.)
# This is the correct way the class variable should be accessed
# print(Workout.cal_per_hr)

# Careful with accessing/modifying the class variable through an instance
# Allowed but don't do it!
# w = Workout('1/1/2021 2:34', '1/1/2021 3:35', None)
# print(w.cal_per_hr)

# Change the class variable value
# Allowed but not good style to change it outside the class definition!
# Workout.cal_per_hr = 250   
# print(w.cal_per_hr) 


# =============================================================================
# TEST: State dictionary for Workout shows class variables too 
# =============================================================================
# print(Workout.__dict__.keys())   # dict_keys(['__module__', '__doc__', 'cal_per_hr', '__init__', 'get_calories', 'get_duration', 'get_start', 'set_calories', 'set_start', 'get_kind', '__eq__', '__str__', '__dict__', '__weakref__', '__hash__'])
# print()
# my_workout = Workout('9/30/2021 1:35 PM', '9/30/2021 1:57 PM', 200)
# print(my_workout.__dict__.keys())  # dict_keys(['start', 'end', 'icon', 'kind', 'calories'])



################## YOU TRY IT #########################
# Create one Workout object saved as variable w_one, 
# from Jan 1 2021 at 3:30 PM until 4 PM. 
# You want to estimate the calories from this workout. 
# Print the number of calories for w_one.
pass

# Create another Workout object saved as w_two, 
# from Jan 1 2021 at 3:35 PM until 4 PM. 
# You know you burned 300 calories for this workout.  
# Print the number of calories for w_two. 
pass

#############################################


# =============================================================================
# EXAMPLE: Subclass of workout to represent a running workout
# =============================================================================

class RunWorkout(Workout):
    
    # new class variable
    cals_per_km = 100
    
    def __init__(self, start, end, elev=0, calories=None, route_gps_points=None):
        """Create a new instance of a running workout, where start and
        end are strings representing the start and end time of the workout,
        and elev is the total elevaation gain in the workout in feet,
        calories is an optional number representing the calories 
        burned in the run, and route_gps_points is an optional array 
        of (lat,lon) pairs representing the route of the run"""
        super().__init__(start,end,calories)
        self.icon = '🏃🏽‍'
        self.kind = 'Running'
        self.elev = elev
        self.route_gps_points = route_gps_points

    def get_elev(self):
        """Return the elevation gain of the workout in feet"""
        return self.elev

    def set_elev(self, e):
        """Sets the elevation gain of the workout in feet, to e"""
        self.elev = e

    def get_calories(self):
        """Return the total calories consumed by the workout, derived
        using 1) the GPS points if supplied, 2) calories, if supplied,
        or 3) an estimate of the calories based on the duration"""
        if (self.route_gps_points != None):
            dist = 0
            lastP = self.route_gps_points[0]
            for p in self.route_gps_points[1:]:
                dist += gpsDistance(lastP,p)
                lastP = p
            return dist * RunWorkout.cals_per_km
        else:
            return super().get_calories()

    def __eq__(self,other):
        """Returns true if this workout is equal to another workout, false o.w."""
        return super().__eq__(other) and self.elev == other.elev


# =============================================================================
# TEST: Inspect internal state of subclasses
# =============================================================================
# print(RunWorkout.__dict__.keys()) # dict_keys(['__module__', 'cals_per_km', '__init__', 'get_elev', 'get_calories', '__eq__', '__doc__', '__hash__'])
# print()
# workout1 = RunWorkout('9/30/2021 1:35 PM','9/30/2021 1:57 PM', calories=450, elev=200)
# print(workout1.__dict__.keys())   # dict_keys(['start', 'end', 'icon', 'kind', 'calories', 'elev', 'route_gps_points'])


# =============================================================================
# EXAMPLE: Subclass of workout to represent a swimming workout
# =============================================================================
class SwimWorkout(Workout):
    """Subclass of workout to representing swimming"""
    
    # redefine class variable cal_per_hr
    cal_per_hr = 400
    
    def __init__(self, start, end, pace, calories=None):
        """Create a new instance of a swimming workout, where start and
        end are strings representing the start and end time of the workout,
        and pace is the pace of the workout in min/100yd, and calories
        is an optional parameter specifying the calories burned in the workout
        """
        super().__init__(start,end,calories)
        self.icon = '🏊‍'
        self.kind = 'Swimming'
        self.pace = pace
    def get_pace(self):
        """Return the pace of the workout"""
        return self.pace
    def get_calories(self):
        """Return the total calories burned in the swim workout
           using the SwimWorkout cal_per_hr class variable"""
        if (self.calories == None):
            # calc the calories based on the length of the workout and cal_per_hr
            return SwimWorkout.cal_per_hr * (self.end - self.start).total_seconds() / 3600.0
        else:
            return self.calories


# =============================================================================
# EXAMPLE:  Show how RunWorkout and SwimWorkout can reuse __str__ from Workout
# =============================================================================

w = Workout('9/30/2021 1:35 PM','9/30/2021 1:57 PM') # uses 200 cal_per_hr
r = RunWorkout('9/30/2021 1:35 PM','9/30/2021 3:57 PM', 100) # uses 200 cal_per_hr
sw = SwimWorkout('9/30/2021 1:35 PM','9/30/2021 1:57 PM', 100) # uses 400 cal_per_hr

# print(w)
# print(r)
# print(sw)


# =============================================================================
# EXAMPLE:  Subclasses can be used in place of workouts
# =============================================================================
def total_calories(workouts):
	cals = 0
	for w in workouts:
		cals += w.get_calories()
	return cals

def total_elevation(run_workouts):
	elev = 0
	for w in run_workouts:
		elev += w.get_elev()
	return elev

w1 = Workout('9/30/2021 1:35 PM','9/30/2021 2:05 PM') # 30 min workout
w2 = Workout('9/30/2021 4:35 PM','9/30/2021 5:05 PM') # 30 min workout
rw1 = RunWorkout('9/30/2021 1:35 PM','9/30/2021 3:35 PM', 100) # 2 hr workout
rw2 = RunWorkout('9/30/2021 1:35 PM','9/30/2021 3:35 PM', 200) # 2 hr workout

# print(total_calories([w1,w2,rw1,rw2]))  # (1) # cal = 100+100+400+400
# print(total_elevation([rw1,rw2]))       # (2) # elev = 100+200
# # total_elevation([w1,rw1])             # (3) # err since w1 has no elev method


############## YOU TRY IT #######################
# Answer the question in the comments
# remember that cal_per_hr is 200
w1 = Workout('9/30/2021 2:20 PM','9/30/2021 2:50 PM')  
# # what is the calories val through get_calories()
# # what is the elevation val through get_elev()
# print(w1.get_calories())
# print(w1.get_elev())

w2 = Workout('9/30/2021 2:20 PM','9/30/2021 2:50 PM', 450)  
# # what is the calories val through get_calories()
# # what is the elevation val through get_elev()
# print(w2.get_calories())
# print(w2.get_elev())

rw1 = RunWorkout('9/30/2021 2:20 PM','9/30/2021 2:50 PM', 250) 
# # what is the calories val through get_calories()
# # what is the elevation val through get_elev()
# print(rw1.get_calories())
# print(rw1.get_elev())

rw2 = RunWorkout('9/30/2021 2:20 PM','9/30/2021 2:50 PM', 450, 700) 
# # what is the calories val through get_calories()
# # what is the elevation val through get_elev()
# print(rw2.get_calories())
# print(rw2.get_elev())

rw3 = RunWorkout('9/30/2021 2:20 PM','9/30/2021 2:50 PM', calories=300) 
# # what is the calories val through get_calories()
# # what is the elevation val through get_elev()
# print(rw3.get_calories())
# print(rw3.get_elev())

#############################################


# =============================================================================
# EXAMPLE:  RunWorkout overrides get_calories() to use GPS points
# Uses lec18_helpers.py to find the distance between (lat, long) pairs
# =============================================================================
# # points are Boston to Newton
points = [(42.3601,-71.0589),(42.3370,-71.2092)] # (latitude,longitude) pairs
run1 = RunWorkout('9/30/2021 1:35 PM','9/30/2021 3:57 PM', 100, route_gps_points=points)
# print(f'Cals with route points: {run1.get_calories()}')

run2 = RunWorkout('9/30/2021 1:35 PM','9/30/2021 3:57 PM', 100) # 100 is elevation
# print(f'Cals with super impl: {run2.get_calories()}')


# =============================================================================
# EXAMPLE: RunWorkout can override class variable cal_per_hr
# =============================================================================
# Create a 1 hour workout with no value for calories
rw = RunWorkout('1/1/2021 1:00 PM', '1/1/2021 2:00 PM', 7.0) # 7 is elevation
# print(rw.get_calories())  # prints 200 since the else calls the parent's get_calories


# =============================================================================
# EXAMPLE: Workouts override __eq__ to provide equality testing
# =============================================================================
w1 = Workout('9/30/2021 1:35 PM','9/30/2021 2:05 PM', 500)
w2 = Workout('9/30/2021 1:35 PM','9/30/2021 2:05 PM') # cal are 200 by default
w3 = Workout('9/30/2021 1:35 PM','9/30/2021 2:05 PM', 100)

rw1 = RunWorkout('9/30/2021 1:35 PM','9/30/2021 3:05 PM', 100)
rw2 = RunWorkout('9/30/2021 1:35 PM','9/30/2021 3:05 PM', 200)
rw3 = RunWorkout('9/30/2021 1:35 PM','9/30/2021 3:05 PM', 100)

# print(w1 == w2)  # False since only length of workout is the same
# print(w1 == w3)  # False since only length of workout is the same
# print(w2 == w3)  # True since the length and calories are equal
# print(w1 == rw1)  # False since the types of w1 and rw1 are not the same 
# print(rw1 == rw2) # False since the elevation is different
# print(rw1 == rw3) # True since the types, workout length, and elev is the same



###################################################################
################## ANSWERS TO YOU TRY IT #########################
###################################################################
# Create one Workout object saved as variable w_one, 
# from Jan 1 2021 at 3:30 PM until 4 PM. 
# The calories from this workout should be calculated by the code. 
# Print the number of calories for w_one.
# w_one = Workout('1/1/2021 3:30 PM', '1/1/2021 4:00 PM')
# print(w_one.get_calories())


# Create another Workout object saved as w_two, 
# from Jan 1 2021 at 3:35 PM until 4 PM. 
# Manually set the calories for this workout to be 300. 
# Print the number of calories for w_two. 
# w_two = Workout('1/1/2021 3:35 PM', '1/1/2021 4:00 PM', 300)
# w_two.set_calories(300)
# print(w_two.get_calories())



#######################################################
############## AT HOME ####################
#######################################################
def total_elapsed_time(L):
    """ L is a list of tuples (e1, e2) where:
        e1 and e2 are strings representing a date and time. e.g. '9/30/2021 1:35 PM'
        e2 occurs later in time than e1
    Consider the elapsed time for a tuple to be the difference between e2 and e1.
    Returns the sum of all the elapsed times, in seconds, in L. """
    pass
    
# t1 = '1/1/2021 2:00 PM'
# t2 = '1/1/2021 2:05 PM'
# t3 = '3/12/2021 1:22 PM'
# t4 = '3/12/2021 1:32 PM'
# t5 = '7/13/2021 6:00 PM'
# t6 = '7/13/2021 6:02 PM'
# L = [(t1, t2), (t3, t4), (t5, t6)]  # 5min + 10min + 2min = 1020 sec
# print(total_elapsed_time(L))    # prints 1020
    
#############################################

#######################################################
# ####### ANSWERS TO AT HOME ################
#######################################################
# def total_elapsed_time(L):
#     total = 0
#     for e in L:
#         # e[0] and e[1]
#         e1 = parser.parse(e[0])
#         e2 = parser.parse(e[1])
#         total += (e2-e1).total_seconds()
#     return total
# #######################################################
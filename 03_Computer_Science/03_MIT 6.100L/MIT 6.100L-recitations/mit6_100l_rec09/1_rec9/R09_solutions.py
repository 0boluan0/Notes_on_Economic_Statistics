from dateutil import parser

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
        self.type = 'Workout'
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

    def set_calories(self, calories):
        """Set the calories of the workout to calories"""
        self.calories = calories

    def set_start(self, start):
        """Set the start time of the workout to the supplied start string"""
        self.start = parser.parse(start)

    def get_type(self):
        """Return the type of the workout as a string"""
        return self.type

    def __eq__(self, other):
        """Returns true if this workout is equal to another workout, false o.w."""
        # the \ breaks up the line
        return type(self) == type(other) and \
                self.start == other.start and \
                self.end == other.end and \
                self.type == other.type and \
                self.get_calories() == other.get_calories()

    def __str__(self):
        """Return a text-based graphical depiction of the workout"""
        width = 16
        retstr =  f"|{'–'*width}|\n"
        retstr += f"|{' ' *width}|\n"
        retstr += f"| {self.icon}{' '*(width-3)}|\n"  #assume width of icon is 2 chars - len('🏃🏽‍♀️');  doesn't do what you'd epxect
        retstr += f"| {self.type}{' '*(width-len(self.type)-1)}|\n"
        retstr += f"|{' ' *width}|\n"
        duration_str = str(self.get_duration())
        retstr += f"| {duration_str}{' '*(width-len(duration_str)-1)}|\n"
        cal_str = f"{round(self.get_calories(),1)}"
        retstr += f"| {cal_str} Calories {' '*(width-len(cal_str)-11)}|\n"

        retstr += f"|{' ' *width}|\n"
        retstr +=  f"|{'_'*width}|\n"

        return retstr


# =============================================================================
# EXAMPLE: Subclass of workout to represent a swimming workout
# =============================================================================
class SwimWorkout(Workout):
    """Subclass of workout to representing swimming"""
    
    # redefine class variable cal_per_hr
    cal_per_hr = 400
    
    def __init__(self, start, end, pace=2, calories=None):
        """Create a new instance of a swimming workout, where start and
        end are strings representing the start and end time of the workout,
        and pace is the pace of the workout in min/100yd, and calories
        is an optional parameter specifying the calories burned in the workout
        """
        super().__init__(start,end,calories)
        self.icon = '🏊‍'
        self.type = 'Swimming'
        self.pace = pace
        self.numlaps = 0
        
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
    
    def get_average_laptime(self):
        if self.numlaps == 0:
            return None
        av_laptime = self.get_duration() / self.numlaps
        return av_laptime
        
    def count_lap(self):
        self.numlaps += 1
    
    def end_swim_workout(self):
        print("Workout Complete")
        print(str(self))
        
    def __str__(self):
        """Return a text-based graphical depiction of the workout"""
        width = 16
        retstr =  f"|{'–'*width}|\n"
        retstr += f"|{' ' *width}|\n"
        retstr += f"| {self.icon}{' '*(width-3)}|\n"  #assume width of icon is 2 chars - len('🏃🏽‍♀️');  doesn't do what you'd epxect
        retstr += f"| {self.type}{' '*(width-len(self.type)-1)}|\n"
        retstr += f"|{' ' *width}|\n"
        duration_str = str(self.get_duration())
        retstr += f"| {duration_str}{' '*(width-len(duration_str)-1)}|\n"
        cal_str = f"{round(self.get_calories(),1)}"
        retstr += f"| {cal_str} Calories {' '*(width-len(cal_str)-11)}|\n"
        numlaps_str = f"{self.numlaps}"
        retstr += f"| {numlaps_str} Num Laps {' '*(width-len(numlaps_str)-11)}|\n"
        
        retstr += f"|{' ' *width}|\n"
        retstr +=  f"|{'_'*width}|\n"

        return retstr
    


# Exercise 1: Modify the Swim Workout Class 
# 1) Create a new attribute for swim workout called numlaps

# 2) Write a new method count_lap which updates numlaps by +1 if called. 

# 3) Write a new method get_average_laptime which calculates the average 
# laptime for a workout. If numlaps is zero, return None. 
# Hint: what does get_duration() return?)

# 4) Overwrite the __str__ dunder method to display Num Laps

# 5) Write a new method end_swim_workout which prints Workout Complete and 
# the Workout summary given by the updated __str__ dunder method




# Exercise 2: Using the New Class 

# 1) Create a new swim workout starting at '9/30/2021 2:20 PM', ending at '9/30/2021 2:50 PM'
rw3 = SwimWorkout('9/30/2021 2:20 PM','9/30/2021 2:50 PM')

# 2) Swim 40 laps
for i in range(40):
    rw3.count_lap()

# 3) Print average laptime
print(rw3.get_average_laptime())

# 4) Display workout summary
rw3.end_swim_workout()
    
    
        



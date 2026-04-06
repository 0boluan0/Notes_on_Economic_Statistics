#################
## HELPER FUNCTION: get the distance between two points
#################
from math import sin, cos, sqrt, atan2, radians

def gpsDistance(p1,p2):
    """Given two points, represented as (lat,lon) tuples, return the
    distance between them in kilometers"""
    R = 6373.0     # approximate radius of earth in km

    lat1 = radians(p1[0])
    lon1 = radians(p1[1])
    lat2 = radians(p2[0])
    lon2 = radians(p2[1])

    #compute haversine distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

    # def __add__(self, other):
    #     newStart = min(self.startDate, other.startDate)
    #     newEnd = max(self.endDate, other.endDate)
    #     newCalories = self.get_calories() + other.get_calories()
    #     return Workout(str(newStart), str(newEnd), newCalories)

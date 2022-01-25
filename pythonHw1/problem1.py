#!/usr/bin/python3
day = 'sunday'
# Your code should be below this line
weekday = {"monday", "tuesday", "wednesday", "thursday", "friday"}
weekend = {"saturday", "sunday"}
if day in weekday:
    print("weekday")
elif day in weekend:
    print("weekend")
else:
    print("neither")





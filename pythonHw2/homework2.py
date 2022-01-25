import math

def histogram(data, n, b, h):
    # data is a list
    # n is an integer
    # b and h are floats
    
    
    # Write your code here
    if n < 0 or h <= b:
        return([])

    if abs(b) > h:
        h = abs(b)
    
    if b < 0:
        b = 0
        for i in range(0, len(data)):
            data[i] = abs(data[i])
    
    hist = [0] * n

    w = (h - b) / n #width of each bin
    
    for h in range(0,n):
        for d in data:
            if d == b: #non inclusive for b
                d = b - 1
            if d >= b + h * w and d < b + (h + 1) * w:
                hist[h] += 1


    # return the variable storing the histogram
    # Output should be a list

    return(hist)


def birthdaycake(name_to_day, name_to_month, name_to_year):
    #name_to_day, name_to_month and name_to_year are dictionaries
    
    # Write your code here
    for name, month in name_to_month.items():
        if month == 10 or month == 11 or month == 12:
            name_to_year[name] += 5

    name_to_all = {}

    for name in name_to_day:
        name_to_all[name] = ((name_to_month[name], name_to_day[name], name_to_year[name]), zeller(name_to_month[name], name_to_day[name], name_to_year[name]))
    


    # return the variable storing name_to_all
    # Output should be a dictionary
    
    return(name_to_all)

def zeller(month, day, year):
    #using zeller formula given in README.md
    daysOfweek = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    m = month
    d = day
    y = year

    if m <= 2:
        m += 12

    if month <= 2:
        y -= 1

    day = (d + math.floor((13 * (m + 1)) / 5) + y + math.floor(y / 4) - math.floor(y / 100) + math.floor(y / 400)) % 7

    day = daysOfweek[day]

    return(day)
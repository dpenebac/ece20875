# Homework 2: Data Structures in Python

### Due Friday, 09.10.2021 at 11:59 PM ET

## Goals

This homework has several objectives:

1. Write some basic Python programs.
2. Get familiar with the different data structures available in Python.
3. Leverage the concept of functions to write modular code.

## Instructions

In this homework, you need to write two Python functions, one per problem described below. Both of these function definitions are provided to you in `homework2.py`. `testhisto.py` and `testname.py` can be used by you to execute your functions in `homework2.py`. We have provided you with some test cases, you may make your own test case and execute to make sure your code runs properly.

### Problem 1

Create a function called `histogram` that takes as input a dataset `data`, a lower bound `b`, an upper bound `h`, and a number of bins `n`, and returns a histogram representation of `data` with `n` bins between these bounds. However, if the lower bound `b` is a negative input, then **set the lower bound `b` to `0` and set the uppebound `h` to the higher value between `h` and the absolute value `b`.** More specifically, your function should:

1. Have input arguments `histogram(data, n, b, h)`, expecting `data` as a list of floats, `n` as an integer, and `b` and `h` as floats.
2. Change `h` to absolute value of `b` if the absolute value of `b` is greater than `h`
3. **if `b < 0` any values in `data` that are negative should be changed to its absolute value.**
4. Change `b` to `0` if `b` is negative.
5. Initialize the histogram `hist` as a list of `n` zeros.
6. Calculate the bin width as `w = (h-b)/n`, so that `hist[0]` will represent values in the range `(b, b + w)`, `hist[1]` in the range `[b + w, b + 2w)`, and so on through `hist[n-1]`. (Remember that `[` is inclusive while `)` is not!). 
7. Ignore any values in `data` that are less than or equal to `b` and greater than or equal to `h`. *Remember if you have change `h` in step 2, you would need to work with the new value of `h`. The same goes for `b`, if modified in step 4*.
8. Increment `hist[i]` by 1 for each value in `data` that belongs to bin `i`, i.e., in the range `[b + i*w, b + (i+1)*w)`.
9. Return `hist`.

At the beginning of your function, be sure to check that `n` is a positive integer and **that `h >= b`**; if not, your code should just return an empty list for `hist`. Please remember to return an empty list. 

For example, typing in

```
data = [-2, -2.2, 0, 5.6, 8.3, 10.1, 30, 4.4, 1.9, -3.3, 9, 8]
hist = histogram(data, 10, -5, 10)
print(hist)
```

should return

```
[0, 1, 2, 1, 1, 1, 0, 0, 2, 1]
```
Some other test cases are:
 
```
data = [-4, -3.2, 0, 7.6, 1.0, 2.2, 30, 2.2, 1.9, -8.3, 6, 5]
hist = histogram(data, 10, -5, 10)
print(hist)
```

should return

```
[0, 2, 2, 1, 1, 1, 1, 1, 1, 0]
```
and,
```
data = [2,2,2]
hist = histogram(data, 3, -2, 3)
print(hist)
```
returns
```
[0, 0, 3]
```
also, 
```
data = [-1,-1,-1,10,10]
hist = histogram(data, 5, -1, 10)
print(hist)
```
returns 
```
[3, 0, 0, 0, 0]
```
Note: Please include all conditions specified in this problem into your code. 

### Problem 2

Create a function called `birthdaycake` that takes as input three dictionaries, `name_to_day`, `name_to_month`, and `name_to_year`, and combines them into a single dictionary `name_to_all` that contains the name of the person as the `key` and the `(month, day, year), weekday` as the `value` of the `name_to_all` dictionary. `weekday` refers to the days of the week, i.e. Monday, Tuesday,..., Sunday. Specifically, your function should:

1. Have input arguments `birthdaycake(name_to_day, name_to_month, name_to_year)`, expecting `name_to_day` as a dictionary mapping a name (string) to a day in the month (integer), `name_to_month` as a dictionary mapping a name (string) to a month (integer)and `name_to_year` as a dictionary mapping a name to a year(integer). You may assume all inputs to be valid.
2. Create a new dictionary `name_to_all` where the keys are all the names contained in `name_to_day`, and contains the person recorded year in the following structure `(month, day, year), weekday`, with `(month, day, year)` being the tuple of values from `name_to_month`, `name_to_day`, and `name_to_year` corresponding to `name`. `weekday` references the day in the week which you need to compute. You may use the [Zeller formula](https://www.codedrome.com/calculating-the-day-of-the-week-with-zellers-congruence-in-python/) or any other resource to calculate the day of the week. **Note**: the *value* we want in this new dictionary is a *tuple*, where the first element of the tuple is a *tuple* of `month, day, year`, and the second element of the tuple is day of the week (string). 
3. Handle the case where a system error caused all people which birth months are October, November, or December to have `5` years deducted from their input.  *You must add `5` years to the year in `(month, day, year), weekday` when month in `(month, day, year), weekday` corresponds to October, November, or December.* Please note: changing the year changes the day of the week as well. 
4. Return `name_to_all`.

For example, typing in

```
name_to_day={'jack':14,'helen':2,'zach':20}
name_to_month={'jack':4,'helen':2,'zach':10}
name_to_year={'jack':2014,'helen':2002,'zach':1969}
```

should return

```
{'jack': ((4, 14, 2014), 'Monday'), 'helen': ((2, 2, 2002), 'Saturday'), 'zach': ((10, 20, 1974), 'Sunday')}
```
*Note the case for `zach`, who has the month of birth set to October. In `zach`'s case, the returned dictionary `name_to_all` has the `year` value changed by adding `5` 1969 to obtain the correct year 1974. This also changes the day of the week `zach` was born in.*

also,
```
name_to_day={'Stive':24,'Bill':28,'Elon':28,'Jeff':12,'Mark':14}
name_to_month={'Stive':2,'Bill':10,'Elon':6,'Jeff':1,'Mark':5}
name_to_year={'Stive':1955,'Bill':1950,'Elon':1971,'Jeff':1964,'Mark':1984}
```
should output

```
{'Stive': ((2, 24, 1955), 'Thursday'), 'Bill': ((10, 28, 1955), 'Friday'), 'Elon': ((6, 28, 1971), 'Monday'), 'Jeff': ((1, 12, 1964), 'Sunday'), 'Mark': ((5, 14, 1984), 'Monday')}
```

*Note the case for `Bill`, who has the month of birth set to November. In `Bill`'s case, the returned dictionary `name_to_all` has the `year` value changed by adding `5` to 1950 to obtain the correct year 1955. This also changes the day of the week `Bill` was born in.*

Note that the specific order you get these elements back may not be the same, because sets and dictionaries do not preserve order. That is OK!

## Testing

We have provided two test programs for you, that recreate the examples from above, in `testhisto1.py`, `testhisto2.py`, `testhisto3.py`,`testhisto4.py`, `testname1.py` and `testname2.py`, which test problems 1 and 2, respectively. Note that these test programs will only work "out of the box" if you have your solution in `homework2.py`. You may verify your code by running the test programs from the terminal. The concept of importing functions from modules or `.py` files are being used here.


## What to Submit

Put the two functions `histogram` and `birthdaycake` in a single file called `homework2`.

Once you have a version of this file (that you have `commit`ted using `git commit` and `push`ed using `git push`) that you are happy with, you are done!
Sit back, relax and enjoy your lectures :)

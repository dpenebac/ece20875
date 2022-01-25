import re
from unittest.signals import registerResult

def problem1(searchstring):
    """
    Match phone numbers.

    :param searchstring: string
    :return: True or False
    """
    #compile a pattern 
    #maybe an area code, maybe 7 numbers maybe 10
    reg = re.compile("(((?=\+)\+(52|1)\s((?=\()\(\d{3}\)\s|\d{3}-))?\d{3}-\d{4})$")
    ans = reg.match(searchstring)
    if ans:
        return(True)
    else:
        return(False)
        
def problem2(searchstring):
    """
    Extract street name from address.

    :param searchstring: string
    :return: string
    """
    reg = re.compile("\d+\s(([A-Z]\w*\s)+)(Ave.|St.|Rd.|Dr.)")
    ans = reg.search(searchstring)
    return(ans.group(1).rstrip() if ans else None)
    
def problem3(searchstring):
    """
    Garble Street name.

    :param searchstring: string
    :return: string
    """
    reg = re.compile("\d+\s(([A-Z]\w*\s)+)(Ave.|St.|Rd.|Dr.)")
    ans = reg.search(searchstring)
    street = ans.group(0) #total strret name (456 yourmomgei ave)

    reg = re.compile("(([A-Z]\w*\s)+)")
    s = reg.search(street).group(1).strip() #street name (yourmomgei)
    rs = s[::-1] #reverse s
    rStreet = reg.sub(rs + ' ', street) #substitute reverse streetname into total streetname

    reg = re.compile("\S*\d+\s(([A-Z]\w*\s)+)\S*") #compile all of string
    ans = reg.sub(rStreet, searchstring)

    return(ans)


if __name__ == '__main__' :
    print(problem1('+1 765-494-4600')) #True
    print(problem1('+52 765-494-4600 ')) #False
    print(problem1('+1 (765) 494 4600')) #False
    print(problem1('+52 (765) 494-4600')) #True    
    print(problem1('494-4600')) #True
    print(problem1(' 494-4600')) #False
    
    print(problem2('The EE building is at 465 Northwestern Ave.')) #Northwestern
    print(problem2('Meet me at 201 South First St. at noon')) #South First
    
    print(problem3('The EE building is at 465 Northwestern Ave.'))
    print(problem3('Meet me at 201 South First St. at noon'))
    #`The EE building is at 465 nretsewhtroN Ave.`
    #`Meet me at 201 tsriF htuoS St. at noon`

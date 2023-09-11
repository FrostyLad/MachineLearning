# Stephen Marsland, 2008

# An exhaustive search to solve the billsfit problem
from numpy import *

def exhaustive():
    maxSize = 500    
    sizes = array([193.71,60.15,89.08,88.98,15.39,238.14,68.78,107.47,119.66,183.70])
    #sizes = array([109.60,125.48,52.16,195.55,58.67,61.87,92.95,93.14,155.05,110.89,13.34,132.49,194.03,121.29,179.33,139.02,198.78,192.57,81.66,128.90])

    best = 0

    twos = arange(-len(sizes),0,1)
    twos = 2.0**twos
    
    for i in range(2**len(sizes)-1):
        string = remainder(floor(i*twos),2) 
        fitness = sum(string*sizes)
        if fitness > best and fitness<500:
            best = fitness
            bestString = string
    print(best)
    print(bestString)
          
exhaustive()

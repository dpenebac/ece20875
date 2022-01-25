import numpy as np
import matplotlib.pyplot as plt
from problem1 import *

data = np.loadtxt('input.txt')
lo = min(data)
hi = max(data)

#print('Uncomment the first segment to verify the histogram you have generated')
histo = plt.hist(data, 10, (lo, hi))[0]
#print(histo)

#print('Uncomment the second segment to verify the normalized histogram you have generated')
#norm_h = norm_histogram(histo)
#print(norm_h)
#print("[0.104, 0.096, 0.094, 0.079, 0.108, 0.092, 0.114, 0.109, 0.121, 0.083]")

#print('Uncomment the third segment to verify the J value you have generated')
#ch = compute_j(plt.hist(data, 5, (lo, hi))[0], (hi - lo) / 5)
#print(ch)
#print("-0.0101")

#print('Uncomment the fourth segment to verify the sweep of J values you have generated')
#ro = sweep_n(data, lo, hi, 1, 100)
#nro = []
#for r in ro:
#    nro.append('%.4f' % r)
#print(nro)

#print('Uncomment the fifth segment to verify the min J score you have generated')
minV = find_min(sweep_n(data, lo, hi, 1, 100))
print(minV)
print("(-0.0101, 0)")

'''
t = linspace(-pi, pi)
omega = 2 * pi / 7
fifth = (sine wave with frequence 5 * omega * t)
k = -2
negative_second = (a sine wave witha frequencey -2 * omega * t)
plot(t, fifth)
plot(t, negative_second) (plot on same graph)
'''

import matplotlib.pyplot as plt
import numpy as np
import math as m

t = np.linspace(-m.pi, m.pi, 100)
omega = 2 * m.pi / 7
k = 5
fifth = np.sin(5 * omega * t)
k = -2
negative_second = np.sin(-2 * omega * t)


figure, axis = plt.subplots(2, 2)
axis[0,0].plot(t, fifth)
axis[0,0].set_title("5 * omega * t")

axis[1,0].plot(t, negative_second)
axis[1,0].set_title("-2 * omega * t")

dt = np.linspace(-3, 3, 100)
axis[0,1].stem(dt, negative_second)
axis[1,1].stem(dt, negative_second)

plt.show()

'''
Fs = 8000
f = 5
sample = 8000
x = np.linspace(sample)
y = np.sin(2 * np.pi * f * x / Fs)
y = np.sin()
plt.plot(x, y)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()
'''
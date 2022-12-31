import numpy as np
from simulator import simulator
import matplotlib.pyplot as plt

n = 3
test = simulator(n)

# Test write register
# test.write(6)
# print(test._register)

# Test read one qbit
# test.had()
# print(test._register)
# print(test.read(2))
# print(test._register)

# Test read all qbits
# N = 2**n
# x = np.arange(N) + 1 
# y = np.zeros(N)

# for i in range(3000):
#     test.reset()
#     test.had()
#     y[test.read()-1] += 1

# plt.bar(x,y)
# plt.show()

# TODO mehr Tests mit Beispielen aus Büchern?
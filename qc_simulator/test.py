from simulator import simulator
from visualization import CircleNotation, DimensionalCircleNotation
import numpy as np

n = 3
sim = simulator(n)

sim.write_integer(0)

# HAD
print('hads')
sim.had([1])
sim.cHad([1], 2)


# CNOTs
print('cnots')
sim.cNot([1],2)
sim.ccNot(1,2,3) # <- wichtig hier int und keine Listen, tut im dasselbe wie sim.cNot([1,2],3), hat aber einen extra Namen (CCNOT bzw. Toffoli)

# SWAP
print('swap')
sim.swap(1, 2)  # <- hier auch in und keine Listen
print('cswap')
sim.cSwap([1], 2, 3) 


# Phase Gates
print('phase')
sim.phase(90, [1])  # winkel (deg) int, liste int oder int
sim.rx(90, [1])
sim.ry(90, [1])
sim.rz(90, [1])

# Controlled Gates
print('cphase')
sim.cPhase([1],2, 90)  # liste int oder int, int, winkel (deg) int
sim.cRx([1],2, 90)
sim.cRy([1],2, 90)
sim.cRz([1],2, 90)

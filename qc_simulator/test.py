from simulator import simulator
from visualization import CircleNotation, DimensionalCircleNotation
import numpy as np

n = 3
sim = simulator(n)
vis = DimensionalCircleNotation(sim)


sim.write_integer(0)

# HAD
print('hads')
sim.had([1])
sim.cHad([1], 2)

vis.show()
vsi.export_png('test.png')



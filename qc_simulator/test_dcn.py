from simulator import simulator
from visualization import CircleNotation, DimensionalCircleNotation
import numpy as np


n = 3
sim = simulator(n)
sim.write_complex([1/np.sqrt(2), 1/np.sqrt(6)*np.exp(-1j*np.pi/4), 0,0,0,0, 0.5, 1/np.sqrt(12)*np.exp(-1j*np.pi/4)])
sim.cNot(1, 2)
sim.had(1)

vis = DimensionalCircleNotation(sim)
# vis.show()

vis.export_png('test.png')
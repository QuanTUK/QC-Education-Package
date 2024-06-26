from src.qc_education_package import Simulator

from src.qc_education_package import DimensionalCircleNotation as Visualization
# from src.qc_education_package import CircleNotation as Visualization

import numpy as np
import matplotlib
matplotlib.use('Tkagg')

# Quantum Key Distribution

# Register as i-th row of comp. basis matrix 
n = 1
sim = Simulator(n)
# sim.writeComplex([0, 1, 1, 0, 0, 0, 0, 0])

vis = Visualization(sim, version=2)

# sim.cSwap([1], 2,3)
vis.show()
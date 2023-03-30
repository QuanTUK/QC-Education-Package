from src.qc_education_package import Simulator, DimensionalCircleNotation
import numpy as np

n = 3
sim = Simulator(n)
vis = DimensionalCircleNotation(sim)

sim.write_integer(0)
sim.had([1])

vis.show()
from simulator import simulator
from visualization import CircleNotation, DimensionalCircleNotation
import numpy as np
import matplotlib, platform


print(f'Python: {platform.python_version()}\nmatplotlib: {matplotlib.__version__}\nnumpy: {np.version.version}\n')

n = 3
sim = simulator(n)
vis = DimensionalCircleNotation(sim, )

# Simulator mit n=[1,2,3] Qubits starten
sim.reset(n)

# Qubits vorbereiten
sim.write(1)  
sim.write_complex([1*np.exp(0)]*2**n)

# Operationen auf Qubits
sim.had(2)
sim.phase(90)
sim.qnot(1)
sim.cNot(1, 2)

# JSON Dump und wiederherstellen 
# dump= str(sim)
# sim2 = simulator(jsonDump=dump)


# zum testen, hat funktioniert
# vis2 = CircleNotation(sim2, cols=2)
vis.show()
# vis2.show()
# QuanTUK Quantum Computing Education Package

This is a python package to get a first glimpse into the general principles of quantum computing.
This toolset may be  used by educators or lectureres to prepare materials for theier classes, and as well interessted students to get first hand-on expierience.

## Installation
For using this package You need Python version 3.9.16 or higher. We recommend installing the latest [python release](https://www.python.org/downloads/).
Having Python installed You can install this toolset using pip
```bash
pip3 install git+https://github.com/QuanTUK/QC-Education-Package.git  
```
## Usage
The following codelines provide a almost minimal examples for using this package 
```python3
from qc_education_package import Simulator, CircleNotation, DimensionalCircleNotation

n = 3  # No. of qubits
sim = Simulator(n)  # Create a quantum computer simulator object
sim.write(1)  # write integer 0 in binary -> |001>
sim.had(1)  # Apply hadamard gate to qubit 0

# for visualizing
sim = DimensionalCircleNotation(sim)
sim.show()
```

For more examples and some interesting insights into the new DCN visualzation we invite You to try the examples provided in the [DCN_Examples](https://github.com/QuanTUK/DCN_examples) Repositroy.

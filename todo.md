# Ziel:
Sehr einfacher Simulator vergleichbar mit der [QCEngine UI](https://oreilly-qc.github.io)
- Modular damit verschiedene Visualisierungen angebaut werden können. UI erbt von Backend
- Methoden angelehnt an [QCEngine](https://oreilly-qc.github.io/docs/build/cheatsheet.html#cheatsheet-label) 
- Kein Noise
- Wie Zustände und Circuit speichern/repräsentieren? 
    - Liste von Operationen
    - Wie machen andere Simulatoren das? (siehe Vergleich/Liste unten)
    - Variablen durch m qbits repräsentieren (wie in QCEngine) noch nicht drin. Brauchen wir das?
- **Aufpassen mit Indexing** -i für korrekte Reihenfolge in Q-Register


## Methoden Todo:
- Zustände erzeugen 
- Methoden? für Gatter. Zuerst die Grundlegenden
    - Single Qubit:
        - ROOTNOT
        - R_xyz Rotation
    - Multi Qubit
        - QFT?

## Universal Quantum Computer
4 Matrizen und 2 Qubits fü universellen Quantncomputer 
- R_x
- R_y
- R_z
- Phase
- C-Not

# Vergleich mit anderen Implementierungen
[Liste mit Simulatoren](https://quantiki.org/wiki/list-qc-simulators). O'Reilly nennt hat die Großen
## [QCEngine UI](https://oreilly-qc.github.io)
- JavaScript
- Methoden Referenz: [CheatSheet](https://oreilly-qc.github.io/docs/build/cheatsheet.html#cheatsheet-label)

## [QISkit](https://qiskit.org)
- [Lernen](https://qiskit.org/learn/) **Mal durchklicken**
- [Docs](https://qiskit.org/documentation/)
- [Github Repo](https://github.com/Qiskit/qiskit)
    - Simulator: [**Qiskit Aer**](https://github.com/Qiskit/qiskit-aer)

## OpenQASM
- [Knowledge Base](https://www.quantum-inspire.com/kbase/cqasm/)
- ursprünglich Quantum Circuit malen
    - Mark up Sprache um Bilder zu erzeugen

## Q#
- Microsoft Azure Quantum
- [EInführung](https://learn.microsoft.com/de-de/azure/quantum/overview-what-is-qsharp-and-qdk)
- [Github Repo](https://github.com/microsoft/qsharp-language)

## Cirq
- Python Modul
- Google Quantum AI
- [Github Repo](https://github.com/quantumlib/cirq)



# Verwendete Numpy Methoden und Docs
## [np.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html)
### @-Operator: [siehe matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul)
### [dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot)
- Wenn 1D, dann inneres Produkt
- Wenn 2D, dann wie matmul

### Matrixmultiplikation: [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)


### Kronecker Produkt: [kron](https://numpy.org/doc/stable/reference/generated/numpy.kron.html)




## [numpy.random](https://numpy.org/doc/stable/reference/random/index.html)
### [rand](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html)
- z.B. für zufälliger Zustand
### [choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)
- Für Messung
- Parameter a: Wenn int dann wird np.arange(a) verwendet, also Liste von 0 bis a-1.
- Array für p muss 1D sein. np.array.flatten() verwenden.

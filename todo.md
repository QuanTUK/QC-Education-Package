# Todo
## Akut
- european journal physics -> einreichen ende märz 31.03 auf arxiv -> 28.29.30 verfübar sein für bugfixes
- text annotationen in tex, Achsen in DCN einzeichnen.
- DCN Version für Webseite fertig machen.
    - settings für bild über kwargs
- **Versionen**:
    - Webseite: Python 3.6.8 , matplotlib 3.3.4, numpy 1.1.95
    - Lokal:    Python 3.10.6, matplotlib 3.7.0, numpy 1.24.2


- Später:
    - Beispiele von QC-Engine übernehmen
    - Namen der Zustände in Binärdarstellung bzw. mehrere Zustände zu Variablen zusammenfassen
    - Branch mit weniger pythonic kram für Education Paper
    - Overleaf für Vorlesung lesen und überarbeiten


## Pipeline
- Plotly Dashboard für Django Webseite
- mehr Tests mit Beispielen aus Büchern?
    - Deutsch und bla algorithmen als funktionen
    - Quantum Bomb Detection
- Phase/Magnitude Logic implementieren
    - Problem Bsp Klausur Risk Managment Pirat
- Visualisierung 
    - Circle Notation implementieren -> Matplotlib
    - Draw Methode für Notation (QISKIT Methode ist openSource)
- Notate anschauen und verwendbar machen
    - MNist (ggf. noch trainieren)
    - Natural Language Model für Physik nutzbar?
- evtl. nicht effizienten Compiler für 1-2 qubits bauen
- PyGame anschauen
    - community Games 
        - avalon
        - debelion -> werwolf und ähnl. spiele
    - Bubble Shooter
    

# Vortrag Projekt
**Termin**: 1 oder 2 Woche Apirl 2023<br>
**Aufpeppen mit coolen Beispielen**
- Tale of Princess and Tiger
- Spiele 
- Info Klassiker, Inc/Dec

# Bachelor 
- Toolbox für Ba wäre soweit fertig
- Komplexe systeme auf was einfaches Reduzieren und damit spaß haben
- Educational andersrum: Bau was das wirklich spaß macht und schau ob Leute damit was lernen



**Minecraft-Mod**
 Spiel -> Mitte Ende nächstes Jahr Master/Promotion nichts für mich

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
- Spielideen? 
    - [Game Based Learning – Computerspiele
in der Hochschullehre](https://www.e-teaching.org/didaktik/konzeption/methoden/lernspiele/game_based_learning/gamebasedlearning.pdf)
    - [Learning by Design: Good Video Games as Learning Machines](https://doi.org/10.2304/elea.2005.2.1.5)
    - Escape Game
    - Game of TUK

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

## [IBM Quantum Composer](https://quantum-computing.ibm.com/composer) 
- [Doks](https://quantum-computing.ibm.com/composer/docs/iqx/visualizations)

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

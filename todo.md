# Todo
## Akut
**Einreichen Ende märz 31.03 auf arxiv -> 28.29.30 verfübar sein für bugfixes**
- Packaging für einfaches installieren auf Collab
- Google Collab .ipynb files als Beispiele:
    - Beispiele QC-Engine
    - Deutsch u.a algorithmen 
    - Phase/Magnitude Logic:
        - Quantum Bomb Detection, Princess Tiger Riddle, Managment Pirat
    - QFT?
    - Info Klassiker, Inc/Dec
- History für Simulator -> zurück Button
- Draw Circuit Option -> Draw Methode QISKIT ist openSource


- settings für bild über kwargs ermöglichen?


## Versionen
    Webseite:   Python 3.6.8
                matplotlib 3.3.4 
                numpy 1.1.95
                
    Lokal:      Python 3.10.6 
                matplotlib 3.7.0
                numpy 1.24.2


## Projekt
- Overleaf für Vorlesung lesen und überarbeiten
- Vortrag bauen und halten


## Pipeline
- effiziente redraw Methode für Animationen
    - Plotly Dashboard für Django Webseite
        - Slider eg für Winkel, Cehckbox für Qubits?
- Notate anschauen und verwendbar machen -> BA
    - MNist (ggf. noch trainieren)
    - Natural Language Model für Physik nutzbar?
- Animationen, die Bits vertauschen bei Operationen, DCN und CN
- PyGame anschauen
    - community Games 
        - avalon
        - debelion -> werwolf und ähnl. spiele
    - Bubble Shooter
- (evtl. nicht effizienten Compiler für 1-2 qubits bauen )


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


# Vergleich mit anderen Implementierungen
[Liste mit Simulatoren](https://quantiki.org/wiki/list-qc-simulators). O'Reilly nennt halt die Großen
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
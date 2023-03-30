import numpy as np
from numpy.random  import rand
import json

class simulator():


    def __init__(self, n=1, jsonDump=None):
        """Constructor for quantum simulator. Creates simulator for n qubit.
        qubit are indexed from 1 to n, registers from 1 to N=2**n.
        It is possible to restore a simulator object with a json dump.

        Args:
            n (int): Number of qubit, defaults to 1.
            jsonDump (str, optional): JSON Dump to restore simulator state from, optional.
        """
        # Prepare qubit base states
        self._zero = np.array([[1],[0]])
        self._one = np.array([[0],[1]])
        # Prepare one qubit gates/matrices
        self._I = np.identity(2)  # Identity in C2
        self._H = 1 / np.sqrt(2) * np.array([[1,1],[1,-1]])  # Hadamard gate
        self._X = np.array([[0,1],[1,0]]) # Not or Pauli X gate
        self._Y = np.array([[0,-1j],[1j,0]]) # Pauli Y gate
        self._Z = np.array([[1,0],[0,-1]]) # Phase 180 or Pauli Z gate
        self._ROOTX = np.array([[1+1j,1-1j],[1-1j,1+1j]])/2 # Phase 180 or Z gate
        self._ROOTZ = np.array([[1,0],[0,1j]]) # Phase 180 or Z gate
        
        self._P  = lambda phi : np.array([[1,0],[0,np.exp(1j * phi)]], dtype=complex) # General phase gate
        self._Rx = lambda phi : np.array([[np.cos(phi/2), -1j*np.sin(phi/2)],[-1j*np.sin(phi/2), np.cos(phi/2)]], dtype=complex) # Rotation about x-axis (Bloch Sphere)
        self._Ry = lambda phi : np.array([[np.cos(phi/2), -1*np.sin(phi/2)],[np.sin(phi/2), np.cos(phi/2),]], dtype=complex) # Rotation about y-axis (Bloch Sphere)
        self._Rz = lambda phi : np.array([[np.exp(-1j*phi/2), 0],[0, np.exp(1j*phi/2)]], dtype=complex) # Rotation about z-axis (Bloch Sphere)

        if jsonDump is not None:
            state = json.loads(jsonDump)
            self._n = int(state['n'])
            amp = np.array(state['amp'])
            phase = np.array(state['phase'])
            self._register = amp*np.exp(1j*phase)
            self._Q_bits = [self._zero] * self._n  # for showing start state in quantum circuit
        else:
            assert(n > 0) 
            self._n = n
            # qubit register
            self._Q_bits = [self._zero] * self._n  # for showing start state in quantum circuit
            self._register = self._nKron(self._Q_bits)
        self._basis = np.identity(2**self._n)
        

    def __str__(self):
        """Overrides objects toString to export simulator in json format.

        Returns:
            str: n and register amplitudes and phases in json format.
        """
        amp = np.abs(self._register).flatten().tolist()
        phase = np.angle(self._register).flatten().tolist()
        return json.dumps({'n': self._n, 'amp': amp, 'phase': phase})


    def reset(self, n=None):
        """Reset simulator to a system of n qubit with all qubit in state |0> (zero state).
        If n is not specified, resets all current qubits to |0>.

        Args:
            n (int, optional): Number of qubits in simulator.
        """
        if n is not None and n != self._n:
            self._n = n
            self._basis = np.identity(2**n)
        self._Q_bits = [self._zero] * self._n  # for showing start state in quantum circuit
        self._register = self._nKron(self._Q_bits)

   
    def write_integer(self, val, qubit=1):  
        """Write given integer value in binary starting a position qubit. If no qubit parameter is passed, start
        with first qubit. Attention! This will override the register. Use this only to prepare your qubit/register.

        Args:
            val (integer): State is build to (1-p)|0> + p|1>. Hence p=0 -> |0>; p=1 -> |1>
            qubit (int or list(int), optional): qubit to be set. Defaults to 1.
        """
        # Check if val can be represented by n_Q_bit qubit
        assert(val < 2**(self._n-qubit+1))
        i = qubit
        bval = np.binary_repr(val, width=self._n) # width is deprecated since numpy 1.12.0, alternative in sim.read
        for d in bval[::-1]:
            self._Q_bits[-i] = self._one if d == '1' else self._zero
            i += 1
        self._register = self._nKron(self._Q_bits)    


    # Alias
    def write(self, val, qubit=1):  
        """Write given integer value in binary starting a position qubit. If no qubit parameter is passed, start
        with first qubit. Attention! This will override the register. Use this only to prepare your qubit/register.
        Alias for qc_simulator.write_integer(val, Q-Bit)

        Args:
            val (float or list): State is build to (1-p)|0> + p|1>. Hence p=0 -> |0>; p=1 -> |1>
            qubit (int or list, optional): qubit(s) to be set. Defaults to None.
        """
        self.write_integer(val, qubit)


    def write_complex(self, a):
        """Prepare register with given complex amplitudes e.g. reg = a0 |0> + a1 |1>
        Attention! This will override the register. Use this only to prepare your qubit/register.

        Args:
            a (list(float)): Complex values for each component of the quantum register.
        """
        assert(len(a)== 2**self._n)
        self._Q_bits = None     # Register not defined by Q_bits
        a = np.array(a, dtype=complex)
        norm = np.linalg.norm(a)
        if abs(norm - 1) > 1e-6:
            print(f"The given amplitudes lead to a not normed state.\nbefore: {np.array2string(np.abs(a), precision=2, floatmode='fixed')}, norm = {norm:2.2f}\nNormalizing...")
            a = a / norm
            print(f"after: {np.array2string(np.abs(a), precision=2, floatmode='fixed')}, norm = {np.linalg.norm(a):2.2f}")
        self._register = np.zeros(len(a), dtype=complex)
        for i in range(len(a)):
            self._register[i] = a[i]


    def write_magn_phase(self, magnitude, phase):
        """Prepare register with given amplitudes and phases e.g. reg = amp0 exp(i phase0) |0> + amp1 exp(i phase1) |1>
        Attention! This will override the register. Use this only to prepare your qubit/register.

        Args:
            absVal (list(float)): abs value for each component 
            phase (list(int)): Integer angle in deg for each component
        """
        assert(len(magnitude) == 2**self._n)
        assert(np.all(np.imag(magnitude) == 0))
        self._Q_bits = None     # Register not defined by Qubits
        magnitude = np.array(magnitude)
        phase = np.array(phase)
        out = magnitude*np.exp(1j * np.deg2rad(phase))
        self.write_complex(out)


    def read(self, qubit=None, basis='c') -> int:
        """Read given qubit. If no qubit is given (qubit=None) all qubit are measured.

        Args:
            qubit (int or list(int), optional): qubit to read. Defaults to None.
            basis (char or np.array, optional): Basis in which measurement ist performed, Either char c: comp. basis; y,x or z: bell basis; h: had; defaults to c 
                                                OR 2x2 np.array to pass custom projector. 

        Returns:
            np.array: Probabilities for for register/single qubit.
        """
        # Set projector for POVM measurement
        if basis == 'c':
            proj = self._I
        elif basis == 'x':
            proj = self._X
        elif basis == 'y':
            proj = self._Y
        elif basis == 'z':
            proj = self._Z
        elif basis == 'h':
            proj = self._H
        elif type(basis) == np.array:
            print("Using custom projector for measurement.")
            # NOTE: expecting 2x2 single qubit operation here. May support suitable matrix, too
            proj = basis

        if qubit is None:
            # Measure all qubits
            if basis != 'c':
                # project all qubit with given projector, POVM Measurement
                self._operatorInBase(proj)
            prop = np.square(np.abs(self._register)).flatten()
            result = np.random.choice(a=2**self._n, p=prop)  # choice uses np.arange 0 to a, hence +1
            self.write_integer(result)
            out = format(result, '0b')
            # Add leading zeros
            if len(out)< self._n:
                out = '0' * (self._n-len(out)) + out
            msg = f"Measured state |{out}>."
        else:
            # Measuring one qubit
            # Measuring using projector to subspace
            if basis != 'c':
                # project all qubit with given projector, POVM Measurement
                self._operatorInBase(proj, qubit)
            # Prop for qubit i in |0> by projection using sp
            pro0 = [np.identity(2)] * self._n 
            # Projective measurement -> POVM
            pro0[-qubit] = self._zero @ self._zero.T # -qubit s.t. order of registers is correct
            pro0 = self._nKron(pro0)
            state0 = pro0 @ self._register
            p0 = np.linalg.norm(state0)
            # Prop for qubit i in |1> by projection using sp
            pro1 = [np.identity(2)] * self._n  
            # Projective measurement -> POVM
            pro1[-qubit] = self._one  @ self._one.T  # -qubit s.t. order of registers is correct
            pro1 = self._nKron(pro1)
            state1 = pro1 @ self._register
            p1 = np.linalg.norm(state1)
            # Check if state was normed
            assert(1 - p0 - p1 < 1e-6)
            # Project to new state
            result = np.random.choice(a=[0,1], p=[p0**2, p1**2])
            state = state1 if result else state0  # True/False is alias for 1/0 in Python 
            norm = p1 if result else p0
            # Normalize state
            self._register = state / norm
            msg =  f"Measurement qubit {qubit:2d}: |{result:d}> \t (|0>: {p0**2:2.2%} |1>: {p1**2:2.2%})."
        print(msg)
        return msg


    #Set global Phase 0
    def setGlobalPhase0(self):
        phase0 = np.angle(self._register[0])
        self._register[0] = np.abs(self._register[0])
        print(f'Phase |000>: {phase0} rad, {np.rad2deg(phase0)} deg')
        return self._operatorInBase(self._P(-phase0)) # NOTE: Das dreht die |1> Zustände


    # Methods to generate single qubit operators
    def had(self, qubit=None) -> np.array:
        """Applies the hadamard gate to given qubit(s).
        If no qubit is given (qubit=None) hadamard gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit(s) to apply HAD to. Defaults to None.

        Returns:
            np.array: Matrix for hadamard gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._H, qubit)
        

    def x(self, qubit=None) -> np.array:
        """Applies the Pauli-X gate to given qubit(s).
        If no qubit is given (qubit=None) NOT gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply NOT to. Defaults to None.

        Returns:
            np.array: Matrix for not gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._X, qubit)


    def y(self, qubit=None) -> np.array:
        """Applies the Pauli-Y gate to given qubit(s).
        If no qubit is given (qubit=None) Y gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply Y to. Defaults to None.

        Returns:
            np.array: Matrix for Y gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._Y, qubit)


    def z(self, qubit=None) -> np.array:
        """Applies the Pauli-Z (PHASE(180) gate to given qubit(s).
        If no qubit is given (qubit=None) NOT gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply Pauli-Z to. Defaults to None.

        Returns:
            np.array: Matrix for Pauli-Z gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._Z, qubit)


    def phase(self, angle, qubit=None):
        """Applies the PHASE gate with given angle (in deg)  to given qubit(s).
        If no qubit is given (qubit=None) PHASE(angle) gate will be applied to all qubits.

        Args:
            angle(int): Angle in deg
            qubit (int or list(int), optional): qubit to apply PHASE to. Defaults to None.

        Returns:
            np.array: Matrix for PHASE gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._P(np.deg2rad(angle)), qubit)


    def rx(self, angle:int, qubit=None):
        """Applies the Rx gate with given angle (in deg)  to given qubit(s).
        If no qubit is given (qubit=None) Rx(angle) gate will be applied to all qubits.

        Args:
            angle(int): Angle in deg
            qubit (int or list(int), optional): qubit to apply Rx to. Defaults to None.

        Returns:
            np.array: Matrix for Rx gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._Rx(np.deg2rad(angle)), qubit)


    def ry(self, angle:int, qubit=None):
        """Applies the Ry gate with given angle (in deg)  to given qubit(s).
        If no qubit is given (qubit=None) Ry(angle) gate will be applied to all qubits.

        Args:
            angle(int): Angle in deg
            qubit (int or list(int), optional): qubit to apply Ry to. Defaults to None.

        Returns:
            np.array: Matrix for Ry gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._Ry(np.deg2rad(angle)), qubit)


    def rz(self, angle:int, qubit=None):
        """Applies the Ry gate with given angle (in deg)  to given qubit(s).
        If no qubit is given (qubit=None) Rz(angle) gate will be applied to all qubits.

        Args:
            angle(int): Angle in deg
            qubit (int or list(int), optional): qubit to apply Rz to. Defaults to None.

        Returns:
            np.array: Matrix for Rz gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._Rz(np.deg2rad(angle)), qubit)


    # Aliases
    def qnot(self, qubit=None) -> np.array:
        """Applies the NOT gate to given qubit(s). Alias for qc_simulator.x(qubit)
        If no qubit is given (qubit=None) NOT gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply NOT to. Defaults to None.

        Returns:
            np.array: Matrix for not gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._X, qubit)


    def flip(self, qubit=None) -> np.array:
        """Applies the Pauli-Z (PHASE(180)) gate to given qubit(s). Alias for qc_simulator.z(qubit) and .phase(180, qubit).
        If no qubit is given (qubit=None) PHASE(90) gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply PHASE(90) to. Defaults to None.

        Returns:
            np.array: Matrix for PHASE(90) gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._P(np.deg2rad(90)), qubit)
    

    def s(self, qubit=None) -> np.array:
        """Applies the relative phase rotation by 90 deg to given qubit(s). Alias for qc_simulator.phase(90, qubit).
        If no qubit is given (qubit=None) PHASE(90) gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply PHASE(90) to. Defaults to None.

        Returns:
            np.array: Matrix for PHASE(90) gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._P(np.deg2rad(90)), qubit)
    

    def t(self, qubit=None) -> np.array:
        """Applies the relative phase rotation by 45 deg to given qubit(s). Alias for qc_simulator.phase(45, qubit).
        If no qubit is given (qubit=None) PHASE(45) gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply PHASE(45) to. Defaults to None.

        Returns:
            np.array: Matrix for PHASE(45) gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._P(np.deg2rad(45)), qubit)


    # Root Gates
    def rootNot(self, qubit=None) -> np.array:
        """Applies the ROOT-NOT (ROOT-X) gate to given qubit(s).
        If no qubit is given (qubit=None) ROOT-NOT gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply ROOT-NOT to. Defaults to None.

        Returns:
            np.array: Matrix for ROOT-NOT gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._ROOTX, qubit)


    def rootZ(self, qubit=None) -> np.array:
        """Applies the ROOT-Z gate to given qubit(s).
        If no qubit is given (qubit=None) ROOT-Z gate will be applied to all qubits.

        Args:
            qubit (int or list(int), optional): qubit to apply ROOT-Z to. Defaults to None.

        Returns:
            np.array: Matrix for ROOT-Z gate on given qubit in comp. basis.
        """
        return self._operatorInBase(self._ROOTZ, qubit)
    

    # Multi qubit gates
    def swap(self, i, j) -> np.array:
        """Performs SWAP operation with given qubit i and j.

        Args:
            i (int): qubit to be swapped.
            j (int): qubit to be swapped.

        Returns:
            np.array: Matrix representation for used SWAP gate in comp. basis.
        """
        # SWAP by using CNOT gates 
        cn1 = self.cNot(i, j)
        cn2 = self.cNot(j, i)
        cn3 = self.cNot(i, j)
        return cn1 @ cn2 @ cn3


    # controlled gates
    def cHad(self, control_qubit, not_Q_bit) -> np.array:
        """Applies the controlled Hadamard gate with given control qubit(s) and target qubit.

        Args:
            control_qubit (int or list(int)): qubit(s) which is controlling.
            not_Q_bit (int): qubit on which Hadamard gate shall be applied.

        Returns:
            np.array: controlled Hadamard gate for given parameters in comp. basis.
        """
        return self._controlledU(self._H, control_qubit, not_Q_bit)


    def cNot(self, control_qubit, not_Q_bit) -> np.array:
        """Applies the CNOT gate with given control qubit(s) and target qubit.

        Args:
            control_qubit (int or list(int)): controlling qubit(s).
            not_Q_bit (int): qubit on which Not gate shall be applied.

        Returns:
            np.array: CNOT gate for given parameters in comp. basis.
        """
        return self._controlledU(self._X, control_qubit, not_Q_bit)


    def ccNot(self, control_qubit1, control_qubit2nt, not_Q_bit) -> np.array:
        """Applies the CCNOT gate with given control qubit(s) and target qubit.

        Args:
            control_qubit1 (int): controlling qubit.
            control_qubit2 (int): controlling qubit.
            not_Q_bit (int): qubit on which Not gate shall be applied.

        Returns:
            np.array: CCNOT gate for given parameters in comp. basis.
        """
        return self._controlledU(self._X, [control_qubit1, control_qubit2], not_Q_bit)


    def cPhase(self, control_qubit, target_Q_bit, angle) -> np.array:
        """Applies the CPHASE gate with given angle in deg, given control qubit(s) and target qubit.

        Args:
            control_qubit (int or list(int)): controlling qubit(s).
            target_Q_bit (int): qubit on which Phase gate shall be applied.
            angle (int): Angle in deg for rotation.

        Returns:
            np.array: CPHASE gate for given parameters in comp. basis.
        """
        return self._controlledU(self._P(np.deg2rad(angle)), control_qubit, target_Q_bit)


    def cRx(self, control_qubit, target_Q_bit, angle) -> np.array:
        """Applies the controlled Rx gate with given angle in deg, given control qubit(s) and target qubit.

        Args:
            control_qubit (int or list(int)): controlling qubit(s).
            target_Q_bit (int): qubit on which Rx gate shall be applied.
            angle (int): Angle in deg for rotation.

        Returns:
            np.array: controlled Rx gate for given parameters in comp. basis.
        """
        return self._controlledU(self._Rx(np.deg2rad(angle)), control_qubit, target_Q_bit)


    def cRy(self, control_qubit, target_Q_bit, angle) -> np.array:
        """Applies the controlled Rx gate with given angle in deg, given control qubit(s) and target qubit.

        Args:
            control_qubit (int or list(int)): controlling qubit(s).
            target_Q_bit (int): qubit on which Ry gate shall be applied.
            angle (int): Angle in deg for rotation.

        Returns:
            np.array: controlled Ry gate for given parameters in comp. basis.
        """
        return self._controlledU(self._Ry(np.deg2rad(angle)), control_qubit, target_Q_bit)


    def cRz(self, control_qubit, target_Q_bit, angle) -> np.array:
        """Applies the controlled Rz gate with given angle in deg, given control qubit(s) and target qubit.

        Args:
            control_qubit (int or list(int)): controlling qubit(s).
            target_Q_bit (int): qubit on which Rz gate shall be applied.
            angle (int): Angle in deg for rotation.

        Returns:
            np.array: controlled Rz gate for given parameters in comp. basis.
        """
        return self._controlledU(self._Rz(np.deg2rad(angle)), control_qubit, target_Q_bit)
    

    def cZ(self, control_qubit, not_Q_bit) -> np.array:
        """Applies the CZ gate with given control qubit(s) and target qubit.

        Args:
            control_qubit (int or list(int)): controlling qubit(s).
            not_Q_bit (int): qubit on which Z gate shall be applied.

        Returns:
            np.array: CZ gate for given parameters in comp. basis.
        """
        return self._controlledU(self._Z, control_qubit, not_Q_bit)


    def cSwap(self, control_qubit, i, j) -> np.array:
        """Performs CSWAP operation with given qubit i and j controlled by given control qubit(s).

        Args:
            control_qubit (int or list(int)): controlling qubit(s).
            i (int): registers to be swapped.
            j (int): registers to be swapped.

        Returns:
            np.array: Matrix representation for used CSWAP gate in comp. basis.
        """
        c1 = [i]
        c2 = [j]
        # NOTE: Das geht schöner
        if type(control_qubit)==list:                
            c1.extend(control_qubit)
            c2.extend(control_qubit)
        elif type(control_qubit)==int:
            c1.append(control_qubit)
            c2.append(control_qubit)
        ccn1 = self.cNot(c2, i)
        ccn2 = self.cNot(c1, j)
        ccn3 = self.cNot(c2, i)
        return ccn1 @ ccn2 @ ccn3
    
    
    # Private/hidden methods
    def _getBasisVector(self, i) -> np.array:
        """Returns i-th basis (row)-vector for dimensions n (comp. basis)

        Args:
            i (int): number of basis vector

        Returns:
            np.array: i-th basis vector (row vector)
        """
        return self._basis[:, i, None]


    def _nKron(self, ops_to_kron) -> np.array:
        """Helper function to apply cascade kroneker products in list

        Args:
            ops_to_kron (list[np.array]): list of matrices to apply in kroneker products

        Returns:
            np.array: Result
        """
        result = 1
        for i in ops_to_kron[::-1]:               
            result = np.kron(i, result)     
        return result


    def _operatorInBase(self, operator, qubit=None) -> np.array:
        """Applies given operator U to given qubit and returns the matrix representation in comp. basis.
        If no qubit is specified (qubit=None) U is applied to all qubits.

        Args:
            operator (np.array): Operator U.
            qubit (int or list(int), optional): Q-Bit on which operator should apply. Defaults to None.

        Returns:
            np.array: Operator U applied to qubit in comp. basis.
        """
        if qubit is None:
            some_list = [operator] * self._n  
        else:
            if type(qubit) == list:
                assert(len(qubit) > 0)
            qubit = np.array(qubit, dtype=int)
            assert(np.all(qubit > 0))
            assert(np.all(qubit <= self._n))
            some_list = np.array([np.identity(2)] * self._n, dtype=complex)
            some_list[-qubit] = operator # -qubit s.t. order of registers is correct 
        op =self._nKron(some_list)
        self._register = op @ self._register
        return op


    def _controlledU(self, operator, control_qubit, target_Q_bit) -> np.array:
        """Returns controlled version of given operator gate

        Args:
            operator (np.array): Matrix form of operator U.
            control_qubit (int or list(int)): Controlling qubit(s).
            target_Q_bit (int): qubit to apply operator to.

        Returns:
            np.array: Matrix for controlled operator in comp. basis.
        """
        control_qubit = np.array(control_qubit, dtype=int)
        assert(np.all(target_Q_bit != control_qubit))
        
        control1 = np.array([np.identity(2)] * self._n, dtype=complex)
        control1[-control_qubit] = np.array([[0,0],[0,1]])  # |1><1| check if |1>

        # NOTE: Test for >3 Qubits!
        # |0><0| check if |0>, apply I if so 
        # For more than one control need to check |0>_i XOR |0>_j  i xor j <=> not(i and j)
        I = self._operatorInBase(self._I)  # I for 2*n
        control0 = I - self._nKron(control1)

        control1[-target_Q_bit] = operator #  apply U if |1><1|
        control1 = self._nKron(control1)  
        
        op = control0 + control1
        self._register = op @ self._register
        return op
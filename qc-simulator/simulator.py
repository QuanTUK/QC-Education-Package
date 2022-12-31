import numpy as np
from numpy.random  import rand

class simulator():

    def __init__(self, n:int):
        """Constructor for quantum simulator. Creates simulator for n qbits.
        Qbits are indexed from 1 to n, registers from 1 to N=2**n

        Args:
            n (int): Number of qbits
        """
        self._n = n
        self._basis = np.identity(2**n)
        # Prepare qbit base states
        self._zero = np.array([[1],[0]])
        self._one = np.array([[0],[1]])
        # Prepare one qbit gates/matrices
        self._I = np.identity(2)  # Identity in C2
        self._H = 1 / np.sqrt(2) * np.array([[1,1],[1,-1]])  # Hadamard gate
        self._X = np.array([[0,1],[1,0]]) # Not or X gate
        self._P = lambda phi : np.array([[1,0],[0,np.exp(1j*phi)]]) # General phase gate
        self._Z = np.array([[1,0],[0,-1]]) # Phase 180 or Z gate
        self._qbits = [self._zero] * n  # for showing start state in quantum circuit
        self._register = self._nKron(self._qbits)


    def reset(self, n=None):
        """Reset simulator to a system of n qbits with all qbits in state |0> (zero state).
        If n is not specified, resets all current qbits to |0>.

        Args:
            n (int, optional): Number of qbits in simulator.
        """
        if n is not None:
            self._n = n
            self._basis = np.identity(2**n)
        self._qbits = [self._zero] * self._n  # for showing start state in quantum circuit
        self._register = self._nKron(self._qbits)


    # Mind! Indexing registers/comp. basis 1 to n
    def write(self, val:int, qbit=1):  
        """Write given integer value in binary starting a position qbit. I no qbit parameter is passed, start
        with first qbit. Attention! This will override the register. Use this only to prepare your qbits/register.

        Args:
            val (integer): State is build to (1-p)|0> + p|1>. Hence p=0 -> |0>; p=1 -> |1>
            qbit (int, optional): qbit to be set. Defaults to 1.
        """
        # Check if val can be represented by n-qbit qbits
        assert(val < 2**(self._n-qbit+1))
        i = qbit
        for d in format(val, 'b')[::-1]:
            self._qbits[-i] = self._one if d=='1' else self._zero
            i += 1
        self._register = self._nKron(self._qbits)


    def set_qbit(self, p:float, qbit=None):  
        """Prepare Qbit i into given state ((1-p)|0> + p|1>). If no qbit is given (qbit=None) prepare all qbits
        to given state. Attention! This will override the register. Use this only to prepare your qbits/register.
        p=0 -> qbit=|0>
        p=1 -> qbit=|1>

        Args:
            p (float): State is build to (1-p)|0> + p|1>. Hence p=0 -> |0>; p=1 -> |1>
            qbit (int, optional): qbit to be set. Defaults to None.
        """
        if qbit is None:
            self._register = self._nKron([(1-p)*self._zero + p * self._one] * self._n)  # this is normed
        else:
            self._qbits[-qbit] = (1-p) * self._zero + p * self._one  # -qbit s.t. order of registers is correct
            self._register = self._nKron(self._qbits)


    def read(self, qbit=None) -> int:
        """Read given qbit. If no qbit is given (qbit=None) all qbits are measured.

        Args:
            qbit (int, optional): Qbit to read. Defaults to None.

        Returns:
            int: Measurement for register/single qbit
        """
        if qbit is None:
            # Measure all qbits
            prop = np.square(self._register).flatten()
            result = np.random.choice(a=2**self._n, p=prop) + 1  # choice uses np.arange 0 to a
        else:
            # Measuring one qbit
            # Measuring using projector to subspace
            # Prop for qbit i in |0> by projection using sp
            pro0 = [np.identity(2)] * self._n 
            pro0[-qbit] = self._zero @ self._zero.T # -qbit s.t. order of registers is correct
            pro0 = self._nKron(pro0)
            state0 = pro0 @ self._register
            p0 = np.linalg.norm(state0)
            # Prop for qbit i in |1> by projection using sp
            pro1 = [np.identity(2)] * self._n  
            pro1[-qbit] = self._one @ self._one.T  # -qbit s.t. order of registers is correct
            pro1 = self._nKron(pro1)
            state1 = pro1 @ self._register
            p1 = np.linalg.norm(state1)
            # Check if state was normed
            assert(1 - p0 - p1 < 1e-6)
            print(f"Measurement qbit {qbit:2d}:\t|0>: {p0**2:2.2%}\t|1>: {p1**2:2.2%}\n")
            # Project to new state
            result = np.random.choice(a=[0,1], p=[p0**2, p1**2])
            state = state1 if result else state0  # True/False is alias for 1/0 in Python 
            norm = p1 if result else p0
            # Normalize state
            self._register = state / norm
        return result


    # Methods to generate single qbit operators
    def had(self, qbit= None) -> np.array:
        """Applies the hadamard gate to qbit i.
        If no qbit is given (qbit=None) hadamard gate will be applied to all qbits.

        Args:
            qbit (int, optional): qbit to apply had to. Defaults to None.

        Returns:
            np.array: Matrix for hadamard gate on given qbit in comp. basis.
        """
        return self._operatorInBase(self._H, qbit)
        

    def qnot(self, qbit=None) -> np.array:
        """Applies the not gate to qbit i.
        If no qbit is given (qbit=None) not gate will be applied to all qbits.

        Args:
            qbit (int, optional): qbit to apply had to. Defaults to None.

        Returns:
            np.array: Matrix for not gate on given qbit in comp. basis.
        """
        return self._operatorInBase(self._X, qbit)


    # Methods to generate multi qbit operators
    def cNot(self, control_qbit:int, not_qbit:int) -> np.array:
        """Applies the CNOT gate with given control and target qbit.

        Args:
            control_reg (int): qbit which is controlling.
            not_qbit (int): qbit on which Not gate shall be applied.

        Returns:
            np.array: c_not gate for given parameters in comp. basis.
        """
        return self._controlledU(self._X, control_qbit, not_qbit)


    def cPhase(self, control_qbit:int, not_qbit:int, angle:int) -> np.array:
        """Applies the CPHASE gate with given angle, given control and target qbit.

        Args:
            control_reg (int): qbit which is controlling.
            not_qbit (int): qbit on which Phase gate shall be applied.
            angle (int): Angle in rad for rotation.

        Returns:
            np.array: CPHASE gate for given parameters in comp. basis.
        """
        return self._controlledU(self._P(angle), control_qbit, not_qbit)

    
    def cZ(self, control_qbit:int, not_qbit:int) -> np.array:
        """Applies the CZ gate with given control and target qbit.

        Args:
            control_reg (int): qbit which is controlling.
            not_qbit (int): qbit on which Z gate shall be applied.

        Returns:
            np.array: CZ gate for given parameters in comp. basis.
        """
        return self._controlledU(self._Z, control_qbit, not_qbit)


    def swap(self, i:int, j:int) -> np.array:
        """Performs SWAP operation with given qbits i and j.

        Args:
            i (int): qbit to be swapped.
            j (int): qbit to be swapped.

        Returns:
            np.array: Matrix representation for used SWAP gate in comp. basis.
        """
        # SWAP by using CNOT gates 
        cn1 = self.cNot(i, j)
        cn2 = self.cNot(j, i)
        cn3 = self.cNot(i, j)
        return cn3 @ cn2 @ cn1


    def cSwap(self, control_qbit:int, i:int, j:int) -> np.array:
        """Performs CSWAP operation with given qbits i and j controlled by given control qbit

        Args:
            control_qbit (int): control qbit.
            i (int): registers to be swapped.
            j (int): registers to be swapped.

        Returns:
            np.array: Matrix representation for used CSWAP gate in comp. basis.
        """
        # CSWAP by using CCNOT gates
        ccn1 = self.cNot([control_qbit, i], j)
        ccn2 = self.cNot([control_qbit, j], i)
        ccn3 = self.cNot([control_qbit, i], j)
        return ccn3 @ ccn2 @ ccn1
    
    
    # Private/hidden methods
    def _getBasisVector(self, i:int) -> np.array:
        """Returns i-th basis (row)-vector for dimensions n (comp. basis)

        Args:
            i (int): number of basis vector

        Returns:
            np.array: i-th basis vector (row vector)
        """
        return self._basis[:, i, None]


    def _nKron(self, ops_to_kron:list) -> np.array:
        """Helper function to apply cascade kroneker products in list

        Args:
            ops_to_kron (list[np.array]): list of matrices to apply in kroneker products

        Returns:
            np.array: Result
        """
        result = 1
        for i in ops_to_kron:
            result = np.kron(result, i)
        return result


    def _operatorInBase(self, operator:np.array, qbit=None) -> np.array:
        """Applies given operator U to given qbit and returns the matrix representation in comp. basis.
        If no qbit is specified (qbit=None) U is applied to all qbits.

        Args:
            operator (np.array): Operator U.
            qbit (int, optional): Q-Bit on which operator should apply. Defaults to None.

        Returns:
            np.array: Operator U applied to qbit in comp. basis.
        """
        if qbit is None:
            some_list = [operator] * self._n
        else:
            some_list = [np.identity(2)] * self._n
            some_list[-qbit] = operator # -qbit s.t. order of registers is correct  
        op = self._nKron(some_list)
        self._register = op @ self._register
        return op


    def _controlledU(self, operator:np.array, control_qbit:int, target_qbit:int) -> np.array:
        """Returns controlled version of given operator gate

        Args:
            operator (np.array): Matrix form of operator U.
            control_qbit (int or list(int)): Controlling qbit(s).
            target_qbit (int): Qbit to apply operator to.

        Returns:
            np.array: Matrix for controlled operator in comp. basis.
        """
        assert(target_qbit not in control_qbit)
        control0 = [np.identity(2)] * self._n
        # Indexing with list should work? TODO check
        control0[-control_qbit] = np.array([[1,0],[0,0]]) # |0><0| check if |0>, apply I if so 
        control1 = [np.identity(2)] * self._n
        control1[-control_qbit] = np.array([[0,0],[0,1]]) # |1><1| check if |1>
        control1[-target_qbit] = operator # apply U if so
        op = self._nKron(control0) + self._nKron(control1)
        self._register = op @ self._register
        return op
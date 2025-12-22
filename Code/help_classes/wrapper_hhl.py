# SciPy core package (used here primarily for linear algebra routines such as matrix exponentials)
import scipy

# PennyLane quantum programming framework (used when package == "PL")
import pennylane              as     qml

# Matplotlib for optional circuit visualization (draw_mpl / circuit plots)
import matplotlib.pyplot      as     plt

# Helper for binary fixed-point / binary-string representations used in control-state construction
from   binary_fractions       import Binary

# NumPy for numerical arrays, eigenvalues, norms, and general linear algebra utilities
import numpy                  as     np

# Qiskit circuit construction + transpilation utilities (used when package == "QS")
from   qiskit                 import QuantumCircuit, transpile

# Qiskit gate primitives used to build controlled unitaries and multi-controlled rotations/phases
from   qiskit.circuit.library import UnitaryGate, CPhaseGate, ZGate, RYGate

# Qiskit Aer simulator backend for ideal/noisy simulation
from   qiskit_aer             import AerSimulator

# Noise model construction (typically derived from a hardware backend) for noisy simulation
from qiskit_aer.noise         import NoiseModel

# Qiskit Runtime Sampler primitive (V2) for submitting circuits (particularly for real-hardware execution)
from   qiskit_ibm_runtime       import SamplerV2 as Sampler


# High-level wrapper implementing an HHL-style workflow across different frameworks/backends:
# - "PL": PennyLane implementations
# - "QS": Qiskit implementations (Aer simulation or IBM Runtime / hardware)
class wrapper_HHL():
    def __init__(self                  ,
                 package               ,
                 modus                 ,       
                 A1                    ,
                 b1                    ,
                 A2             = 0    ,
                 b2             = 0    ,
                 simulator      = ""   ,
                 service        = ""   ,
                 backend        = ""   ,
                 transpile      = "sim",
                 flag_use_gpu   = False,
                 flag_real_qc   = None ,
                 print_solution = True ,
                 print_circuit  = False,
                 n_shots        = 10000,
                 factor_t       = 0.99 ):

        # Mode selector controlling which workflow variant is executed (e.g., 1xA_1xb, etc.)
        self.modus              = modus

        # Backend/framework selector: "PL" for PennyLane, "QS" for Qiskit
        self.package            = package

        # Simulator mode selector (e.g., "ideal", "noisy"); interpretation depends on package
        self.simulator          = simulator

        # Primary linear system definition A1 x = b1
        self.A1                 = A1
        self.A2                 = A2
        self.b1                 = b1
        self.b2                 = b2

        # Number of phase-estimation / eigenvalue register qubits (fixed here; used in Hamiltonian simulation & QFT)
        self.nl2                = 8

        # Qiskit Runtime service handle (required for hardware/noise-model backends in QS path)
        self.service            = service

        # Backend name (e.g., "ibm_aachen") used for transpilation and/or execution in QS path
        self.backend            = backend

        # Transpilation target:
        # - "sim": transpile for simulator backend
        # - "qc":  transpile for a specific hardware backend
        self.transpile          = transpile

        # GPU acceleration toggle (used in QS path with AerSimulator, or PL lightning.gpu device)
        self.flag_use_gpu       = flag_use_gpu

        # Execution toggle:
        # - True: run on real quantum hardware (QS Runtime)
        # - False: run in simulator (ideal/noisy depending on configuration)
        # - None:  allow caller to omit and infer behavior from simulator flag
        self.flag_real_qc       = flag_real_qc

        # Whether to print detailed numerical results (ratios, norms, etc.) for each run
        self.print_solution     = print_solution

        # Whether to draw/plot the circuit (framework-dependent)
        self.print_circuit      = print_circuit

        # Number of shots used for sampling-based execution
        self.n_shots            = n_shots

        # Scaling factor used when computing t (Hamiltonian simulation time) relative to the theoretical bound
        self.factor_t           = factor_t



    def e_iAt(self, A, t):

        # Matrix exponential exp(i A t); used to build controlled time-evolution unitaries for phase estimation
        return scipy.linalg.expm(t*A*1.j)


    def get_t(self, lambdas):

        # Choose whether to use a 2π periodicity factor based on whether all eigenvalues are positive
        # (if all eigenvalues are positive, a factor of 2 is used; otherwise 1)
        factor_pi = 2 if np.sum(lambdas > 0) == self.nl else 1

        # Scale evolution time inversely with the maximum eigenvalue magnitude to keep phases within a target range
        return (self.factor_t*factor_pi*np.pi) / np.max(np.abs(lambdas))


    def get_lambda_tilde(self, lmbd, t):

        # Map eigenvalue λ to the fixed-point phase-estimation representation λ~ (scaled by 2^nl2)
        # Negative λ values are wrapped into [0,1) via modulo arithmetic
        if lmbd > 0:
            return ((2**self.nl2)*   lmbd*t)/(2*np.pi)
        else:
            return  (2**self.nl2)*(((lmbd*t)/(2*np.pi)) % 1)


    def get_C(self, lambdas_tilde):

        # Choose a constant C slightly below min(λ~) to keep controlled rotations well-defined (C/λ~ <= 1)
        return 0.99999999999999999*min(lambdas_tilde)


    def get_rotation_angle(self, C, lambda_tilde):

        # Controlled-rotation angle used in HHL: 2*arcsin(C/λ~)
        return 2 * np.arcsin(C/lambda_tilde)


    def State_Preparation(self, b, wires):

        # Keep track of sign pattern separately; state amplitudes are prepared from absolute values
        signs = np.sign(b)
        b     = np.abs( b)

        # Ensure wires are in NumPy array form for indexing convenience
        wires = np.array(wires)

        j     = 1

        # First-level amplitude tree:
        # beta_enum / beta_denom defines the first rotation on the most significant qubit
        # beta: enumerator
        beta_enum  = np.sqrt(np.sum([b[(2*j-1)*2**(self.nb-1)+l]**2 for l in range(2**(self.nb-1))]))

        # beta: denominator
        beta_denom = np.sqrt(np.sum([b[(  j-1)*2**(self.nb  )+l]**2 for l in range(2**self.nb    )]))

        # calculate rotation angle for top-level split of probability mass
        if beta_enum == 0:
            beta = 0
        else:
            beta = 2*np.arcsin(beta_enum/beta_denom)

        # Apply the initial Y rotation (framework-dependent)
        if   self.package == "PL":
            qml.RY(phi   = beta    ,
                    wires = wires[0])
        elif self.package == "QS":
            self.qc.ry(beta, wires[0])

        # Recursively encode amplitudes via a binary-tree of controlled RY rotations
        for s in reversed(range(1, self.nb)):
            for j in reversed(range(1, 2**(self.nb-s)+1)):

                # beta: enumerator for the current subtree
                beta_enum  = np.sqrt(np.sum([b[(2*j-1)*2**(s-1)+l]**2 for l in range(2**(s-1))]))

                # beta: denominator for the current subtree
                beta_denom = np.sqrt(np.sum([b[(  j-1)*2**(s  )+l]**2 for l in range(2**s    )]))

                # calculate conditional rotation angle for this subtree
                if beta_enum == 0:
                    beta   = 0
                else:
                    beta   = 2*np.arcsin(beta_enum/beta_denom)

                
                # Determine which wires act as controls and which wire is the current target in the tree
                wires          = list(wires)
                control_wires  = wires[: self.nb - s]
                target         = wires[  self.nb - s]

                # Control pattern corresponds to the binary index (j-1) over the control register
                control_values = [int(l) for l in np.binary_repr(num   = j-1      ,
                                                                 width = self.nb-s)]
                
                
                if   self.package == "PL":
                                        
                    # PennyLane: apply multi-controlled RY with explicit control_values pattern
                    qml.ctrl(op             = qml.RY        ,
                             control        = control_wires ,
                             control_values = control_values)(
                             phi            = beta          ,
                             wires          = target        )
                
                elif self.package == "QS":
                    
                    # Qiskit: build ctrl_state bitstring matching the control pattern
                    control_values = "".join(str(x) for x in control_values)

                    # Build an n-controlled RY gate conditioned on the given ctrl_state
                    mc_ry          = RYGate(beta).control(num_ctrl_qubits = len(control_wires),
                                                            ctrl_state      = control_values    )

                    # Append with Qiskit qubit ordering (controls first, then target)
                    self.qc.append(mc_ry, control_wires[::-1] + [target])

        # Apply sign corrections: implement phase flips on basis states corresponding to negative entries in b
        for k, s in enumerate(signs):
            if s < 0:

                # Binary label for the basis state |k> in nb qubits
                bits   = format(k, f"0{self.nb}b")
                target = wires[-1]

                # If LSB is 0, flip target to convert a controlled-Z into the desired pattern on the last qubit
                if bits[-1] == "0":
                    if   self.package == "PL":
                        qml.PauliX(wires = target)
                    elif self.package == "QS":
                        self.qc.x(         target)

                # Controls are all but the last wire; pattern is bits[:-1]
                wires          = list(wires)
                control_wires  = wires[:-1]
                control_values = [int(b) for b in bits[:-1]]

                if   self.package == "PL":
                    # PennyLane: multi-controlled Z phase flip on the target for the specified control pattern
                    qml.ctrl(op             = qml.PauliZ    ,
                             control        = control_wires ,
                             control_values = control_values)(
                             wires          = target        )
                    
                elif self.package == "QS":

                    # Qiskit: build ctrl_state bitstring (pattern of required control qubit values)
                    control_values  = "".join(str(v) for v in control_values)

                    # Multi-controlled Z (phase flip) gate conditioned on ctrl_state
                    mc_z            = ZGate().control(num_ctrl_qubits = len(control_wires),
                                                      ctrl_state      = control_values    )

                    # Append (controls first, then target) with the chosen ordering convention
                    self.qc.append(mc_z, control_wires[::-1] + [target])
                

                # Undo the earlier X if it was applied (restore computational basis)
                if bits[-1] == "0":
                    
                    if   self.package == "PL":
                        qml.PauliX(wires = target)
                    elif self.package == "QS":
                        self.qc.x(         target)
        
        # Barrier for readability / separation of stages (framework-dependent)
        if self.package == "PL":
            qml.Barrier()
        elif self.package == "QS":
            self.qc.barrier()


    def Hadamards(self, start):
        # Apply Hadamard gates to the phase-estimation register (length nl2) starting at wire index `start`
        # This creates the uniform superposition required for Quantum Phase Estimation (QPE).
        for i in range(start, start+self.nl2):
            if   self.package == "PL":
                qml.Hadamard(wires = i)
            elif self.package == "QS":
                self.qc.h(i)

        # Insert a barrier to visually/structurally separate algorithmic stages in the circuit
        if   self.package == "PL":
            qml.Barrier()
        elif self.package == "QS":
            self.qc.barrier()


    def Hamiltonian_Simulation(self, A, t):

        # Target wires for system register (the register encoding |b> / |x>), assumed to be the first nb wires
        wires = list(range(self.nb))

        # Controlled time-evolution unitaries for QPE:
        # apply controlled-U^(2^x) (here: exp(i A t 2^x)) for each phase qubit x
        for x in range(0, self.nl2):

            # Single control wire for the x-th phase qubit (offset by nb)
            control_wires     = [self.nb+x]

            # Control state "1" (apply unitary only when the control qubit is |1>)
            control_values    = [1]

            # Time-evolution unitary U = exp(i A t 2^x)
            U                 = self.e_iAt(A, (t*(2**x)))

            if   self.package == "PL":

                # PennyLane: apply controlled QubitUnitary with explicit control_values
                qml.ctrl(op             = qml.QubitUnitary,
                         control        = control_wires   ,
                         control_values = control_values  )(
                         U              = U               ,
                         wires          = wires           )

            elif self.package == "QS":

                # Qiskit: build the bitstring for ctrl_state (here always "1" for a single control)
                control_values = "".join(str(bit) for bit in control_values)

                # Wrap the matrix U as a Qiskit UnitaryGate
                base_gate      = UnitaryGate(data  =  U ,
                                             label = "U")

                # Create the controlled version of U with the specified control state
                mcU            = base_gate.control(num_ctrl_qubits = len(control_wires),
                                                   ctrl_state      = control_values    )

                # Append with the chosen ordering convention (controls first, then target wires)
                self.qc.append(mcU, control_wires[::-1] + wires[::-1])

        # Barrier to separate the Hamiltonian simulation block in the circuit diagram
        if   self.package == "PL":
            qml.Barrier()
        elif self.package == "QS":
            self.qc.barrier()


    def Hamiltonian_Simulation_inverse(self, A, t):

        # Target wires for the system register (same convention as forward simulation)
        wires = list(range(self.nb))

        # Apply the inverse of the controlled time-evolution unitaries in reverse order
        # (i.e., uncompute the QPE entanglement after controlled rotations)
        for x in range(self.nl2-1, -1, -1):

            # Single control wire for the x-th phase qubit (offset by nb)
            control_wires  = [self.nb+x]

            # Control state "1"
            control_values = [1]

            # Inverse time-evolution: U^\dagger = exp(-i A t 2^x)
            U              = self.e_iAt(A, -(t*(2**x)))

            if   self.package == "PL":

                # PennyLane: controlled QubitUnitary for the inverse evolution
                qml.ctrl(op             = qml.QubitUnitary,
                         control        = control_wires   ,
                         control_values = control_values  )(
                         U              = U               ,  
                         wires          = wires           )


            elif self.package == "QS":

                # Qiskit: ctrl_state bitstring (single-bit string "1")
                control_values = "".join(str(bit) for bit in control_values)

                # Wrap U as a UnitaryGate
                base_U         = UnitaryGate(data  =  U ,
                                             label = "U")

                # Controlled-U with the specified control state
                mcU            = base_U.control(num_ctrl_qubits = len(control_wires),
                                                ctrl_state      = control_values    )

                # Append controlled inverse evolution (controls first, then target wires)
                self.qc.append(mcU, control_wires[::-1] + wires[::-1])


        # Barrier to separate the inverse Hamiltonian simulation block
        if   self.package == "PL":
            qml.Barrier()
        elif self.package == "QS":
            self.qc.barrier()


    def QFT(self, start):

        # Quantum Fourier Transform on the phase-estimation register starting at wire index `start`
        # Implements the standard QFT circuit with Hadamards and controlled phase rotations.
        for k in range(start, start+self.nl2, +1):

            # Apply Hadamard to the current QFT wire
            if   self.package == "PL":
                qml.Hadamard(wires = [k])
            elif self.package == "QS":
                self.qc.h(            k )
            
            # Apply controlled phase shifts from later wires j onto current wire k
            for j in range(k+1, start+self.nl2, +1):

                wires = [j, k]

                # Phase angle for QFT controlled rotations
                phi   = np.pi/(2**((j-k)))

                if   self.package == "PL":
                    qml.ControlledPhaseShift(phi    = phi , wires = wires)
                elif self.package == "QS":
                    self.qc.append(CPhaseGate(theta = phi),         wires)

            # Barrier after each QFT layer for readability/debugging
            if   self.package == "PL":
                qml.Barrier()
            elif self.package == "QS":
                self.qc.barrier()


    def QFT_inverse(self, start):

        # Inverse Quantum Fourier Transform (QFT†) on the phase-estimation register starting at `start`
        # Implemented by reversing the QFT gate order and negating the controlled-phase angles.
        for k in reversed(range(start, start+self.nl2, +1)):           

            # Apply controlled phase shifts in reverse order (undoing QFT entanglement)
            for j in reversed(range(k+1, start+self.nl2, +1)):

                wires = [j, k]

                # Negative angle implements the inverse of the QFT controlled rotation
                phi   = -np.pi/(2**(j-k))

                if   self.package == "PL":
                    qml.ControlledPhaseShift(phi    = phi , wires = wires)
                elif self.package == "QS":
                    self.qc.append(CPhaseGate(theta = phi),         wires)

            # Finish the inverse-QFT layer with a Hadamard on wire k
            if   self.package == "PL":
                qml.Hadamard(wires = [k])
                qml.Barrier()
            elif self.package == "QS":
                self.qc.h(            k )
                self.qc.barrier()


    def Controlled_RY(self, lambdas_tilde, rotation_angles):

        # Target wire for the ancilla rotation (placed after system register and phase register)
        wires         = self.nb + self.nl2

        # Control wires correspond to the phase-estimation register (nb .. nb+nl2-1)
        control_wires = list(range(self.nb, self.nb+self.nl2))

        def cry_cz(package, control_wires, control_values, rotation_angle, target):

                    # Implement a multi-controlled RY on `target` conditioned on `control_values`,
                    # and (optionally) a controlled-Z to encode the sign when rotation_angle < 0.
                    angle              = abs(rotation_angle)

                    if   package == "PL":
                        
                        # PennyLane expects control_values as a list of ints (not a bitstring)
                        control_values = [int(x) for x in control_values]

                        # Multi-controlled RY with specified control pattern
                        qml.ctrl(op             = qml.RY        ,
                                 control        = control_wires ,
                                 control_values = control_values)(
                                 phi            = angle         ,
                                 wires          = target        )

                    elif package == "QS":
                        
                        # Qiskit expects ctrl_state as a bitstring encoding the control pattern
                        mc_ry      = RYGate(angle).control(num_ctrl_qubits = len(control_wires),
                                                           ctrl_state      = control_values    )

                        # Append controlled rotation (controls first, then target) with the chosen ordering convention
                        self.qc.append(mc_ry, control_wires[::-1] + [target])


                    # If the original rotation angle is negative, encode the sign via an additional controlled-Z phase flip
                    if rotation_angle < 0:

                        if   package == "PL":
                            qml.ctrl(op             = qml.PauliZ    ,
                                     control        = control_wires ,
                                     control_values = control_values)(
                                     wires          = target        )

                        elif package == "QS":

                            # Multi-controlled Z conditioned on the same ctrl_state pattern
                            mc_z        = ZGate().control(num_ctrl_qubits = len(control_wires),
                                                          ctrl_state      = control_values    )

                            # Append the phase flip (controls first, then target)
                            self.qc.append(mc_z, control_wires[::-1] + [target])


        # For each eigenvalue entry (discretized), apply controlled rotations for nearby integer bins
        # (lambda_temp-1, lambda_temp, lambda_temp+1, lambda_temp+2) to reduce discretization artifacts.
        for x in range(self.nl):

            if   self.package == "PL" or self.package == "QS":

                # Discretize λ~ to an integer bin used to build control patterns for the phase register
                lambda_temp = int(lambdas_tilde[x])

                # Build several neighboring control-state bitstrings (binary fixed-point representations)
                cvalues0 = str(Binary(lambda_temp-1)).split(".")[0][2:]
                cvalues1 = str(Binary(lambda_temp  )).split(".")[0][2:]
                cvalues2 = str(Binary(lambda_temp+1)).split(".")[0][2:]
                cvalues3 = str(Binary(lambda_temp+2)).split(".")[0][2:]

                # Left-pad to nl2 bits to match the size of the phase register
                if len(cvalues0) <   self.nl2:
                    cvalues0  = "0"*(self.nl2-len(cvalues0)) + cvalues0
                if len(cvalues1) <   self.nl2:
                    cvalues1  = "0"*(self.nl2-len(cvalues1)) + cvalues1
                if len(cvalues2) <   self.nl2:
                    cvalues2  = "0"*(self.nl2-len(cvalues2)) + cvalues2
                if len(cvalues3) <   self.nl2:
                    cvalues3  = "0"*(self.nl2-len(cvalues3)) + cvalues3

                # Apply the controlled (RY + optional CZ) for each nearby bin using the same rotation angle
                cry_cz(package        = self.package      ,
                       control_wires  = control_wires     ,
                       control_values = cvalues0          ,
                       rotation_angle = rotation_angles[x],
                       target         = wires             )

                cry_cz(package        = self.package      ,
                       control_wires  = control_wires     ,
                       control_values = cvalues1          ,
                       rotation_angle = rotation_angles[x],
                       target         = wires             )

                cry_cz(package        = self.package      ,
                       control_wires  = control_wires     ,
                       control_values = cvalues2          ,
                       rotation_angle = rotation_angles[x],
                       target         = wires             )

                cry_cz(package        = self.package      ,
                       control_wires  = control_wires     ,
                       control_values = cvalues3          ,
                       rotation_angle = rotation_angles[x],
                       target         = wires             )


        # Barrier to separate the controlled-rotation stage in the circuit diagram
        if   self.package == "PL":
            qml.Barrier()
        elif self.package == "QS":
            self.qc.barrier()


    def get_params(self):

        def check_complex(b, text):
             # Guardrail: HHL implementation here assumes real-valued input vector b (complex entries are rejected)
            if np.all(np.isreal(b)) == False:
                print("At least one element has a non-zero imaginary part in " + text + ".")

                # Signal that an invalid (complex) input was detected
                return True
            else:
                # Input is purely real-valued
                return False


        # Abort parameter setup if b1 contains complex entries
        if check_complex(self.b1, "b1"):
            return None

        # Normalize b1 and store both the normalization factor and the normalized vector
        self.b1_norm_factor = np.linalg.norm(x = self.b1)

        self.b1_norm        = self.b1 / self.b1_norm_factor

        # Determine number of system qubits (nb) from the dimension of b1 (expects power-of-two length)
        self.nb          = int(np.log2(len(self.b1)))

        # Convenience: system dimension (nl) equals vector length (= 2**nb)
        self.nl          = len(self.b1)

        # self.nl2         = 8

        # Compute eigenvalues of A1 (used for scaling, time selection, and rotation-angle computation)
        lambdas1         = np.sort(np.linalg.eig(self.A1)[0])[::-1]

        # Keep an unnormalized copy of A1 for classical reference solution
        self.A1_unnormed = self.A1

        # Normalize A1 if necessary so that max |λ| <= 1 (improves stability of time-evolution scaling)
        if max(abs(lambdas1)) > 1:
            self.A1_factor = max(abs(lambdas1))
            self.A1        = self.A1  / self.A1_factor
            lambdas1       = lambdas1 / self.A1_factor
        else:
            # No rescaling required
            self.A1_factor = 1

        # Compute condition number κ(A1) = max|λ| / min|λ| (based on normalized eigenvalues)
        self.condtional_number1 = max(np.abs(lambdas1)) / min(np.abs(lambdas1))
        
        # Choose Hamiltonian simulation time t1 from the spectrum (scaled by factor_t)
        self.t1                 = self.get_t(lambdas = lambdas1)

        # Compute discretized eigenvalue representation λ~ for phase-estimation postprocessing
        self.lambdas1_tilde     = []

        for x in range(self.nl):
            self.lambdas1_tilde.append(self.get_lambda_tilde(lmbd = lambdas1[x], t = self.t1))

        # Choose C parameter for controlled rotation (kept slightly below the minimum to satisfy C/λ <= 1)
        C1 = self.get_C(lambdas_tilde = np.abs(lambdas1))

        # Compute controlled-rotation angles used in the HHL ancilla rotation step
        self.rotation_angles1 = []
        for x in range(self.nl):
            self.rotation_angles1.append(self.get_rotation_angle(C1, lambdas1[x]))

        # Classical reference solution for later comparison / sign inference / rescaling
        self.x_c1 = self.get_solution_classic(self.A1_unnormed, self.b1)
        

    def get_counts(self):

        # Populate all derived parameters (normalizations, eigenvalues, t, angles, reference solution)
        self.get_params()

        # Total circuit wires:
        # - nb system qubits
        # - nl2 phase-estimation qubits
        # - 1 ancilla qubit (for controlled rotation / postselection)
        wires_circuit     = self.nb + self.nl2 + 1

        # Measurement wires:
        # - system register (nb wires)
        # - ancilla wire at index (nb+nl2)
        wires_measurement = list(range(self.nb)) + [self.nb+self.nl2]
         
        def qnode_1xA_2xA_1xb_2xb():

            # Prepare |b> on the system register
            self.State_Preparation(b     = self.b1_norm  ,
                                   wires = range(self.nb))

            # Create uniform superposition on the phase register (QPE initialization)
            self.Hadamards(start = self.nb)
            
            # Apply controlled time-evolution exp(i A t 2^k) for QPE
            self.Hamiltonian_Simulation(A = self.A1,
                                        t = self.t1)

            # Apply inverse QFT to convert accumulated phases into computational basis
            self.QFT_inverse(start = self.nb)

            # Apply eigenvalue-dependent controlled rotations on the ancilla (HHL "inversion" step)
            self.Controlled_RY(lambdas_tilde   = self.lambdas1_tilde  ,
                                rotation_angles = self.rotation_angles1)
            
            # Reapply QFT and inverse Hamiltonian simulation to uncompute the phase register
            self.QFT(start = self.nb)

            self.Hamiltonian_Simulation_inverse(A = self.A1,
                                                t = self.t1)

            # Final Hadamards to complete the uncomputation of the phase register
            self.Hadamards(start = self.nb)
            
        
        if self.package == "PL":

            # PennyLane execution path (used when package == "PL")
            if self.simulator == "ideal":

                # Select PennyLane device backend for ideal simulation.
                # - lightning.gpu: faster statevector simulation on supported GPU setups (requires Lightning + GPU support)
                # - default.qubit: portable CPU fallback (works on standard installations; slower but broadly reproducible)
                if self.flag_use_gpu:
                    dev = qml.device("lightning.gpu"      ,
                                     wires = wires_circuit,
                                     shots = self.n_shots )
                else:
                    dev = qml.device("default.qubit"      ,
                                     wires = wires_circuit,
                                     shots = self.n_shots )
                

                # Wrap the circuit-building routine into a PennyLane QNode bound to the selected device.
                # This produces a callable function that executes the quantum program and returns measurement statistics.
                @qml.qnode(dev)
                def qnode_1xA_2xA_1xb_2xb_highlevel():
                    # Build the circuit (preparation + QPE + controlled rotation + uncomputation)
                    # and return shot-based measurement counts on the chosen wires.
                    qnode_1xA_2xA_1xb_2xb()

                    return qml.counts(wires        = wires_measurement,
                                      all_outcomes = True             )
                
                # Expose the high-level QNode as `qnode` for downstream execution and optional drawing
                qnode = qnode_1xA_2xA_1xb_2xb_highlevel

            # Optional circuit drawing (Matplotlib)
            if self.print_circuit:
                plt.close("all")

                fig, ax = qml.draw_mpl(qnode)()
                fig.show()

            # Execute the circuit and store counts
            self.counts = qnode()
        

        elif self.package == "QS":
          
            # Create a Qiskit QuantumCircuit:
            # - wires_circuit quantum bits
            # - (nb + 1) classical bits for measurement of system register and ancilla
            self.qc = QuantumCircuit(wires_circuit,
                                     self.nb + 1  )

            # Populate self.qc by running the shared circuit-construction routine
            qnode_1xA_2xA_1xb_2xb()

            # Measure system register and ancilla into classical bits (note the reversed wire order)
            self.qc.measure(wires_measurement[::-1], range(len(wires_measurement)))

            if self.simulator == "ideal":
               
                # Ideal Aer simulation using statevector method (optionally GPU-accelerated)
                if self.flag_use_gpu:
                    sim         = AerSimulator(method = "statevector",
                                               device = "GPU"        )
                else:
                    sim         = AerSimulator(method = "statevector")

                # Ensure transpilation targets simulator when running in ideal mode
                self.transpile  = "sim"


            elif self.simulator == "noisy" or self.flag_real_qc:
                
                # Retrieve the target hardware backend (used either for noise model or for real execution)
                backend_hw      = self.service.backend(name = self.backend)

                if not self.flag_real_qc:
                    
                    # Build an Aer noise model derived from the hardware backend calibration data
                    noise_model = NoiseModel.from_backend(backend_hw)

                    # Create an AerSimulator configured to mirror the backend topology + basis gates + noise
                    if self.flag_use_gpu:
                        sim = AerSimulator.from_backend(backend     = backend_hw ,
                                                        noise_model = noise_model,
                                                        device      = "GPU"      )
                    else:
                        sim = AerSimulator.from_backend(backend     = backend_hw ,
                                                        noise_model = noise_model)
                        
                    # When simulating noise, transpile against the simulator backend
                    self.transpile = "sim"

            
            # Choose the transpilation target:
            # - hardware backend if transpile=="qc" or if executing on real hardware
            # - simulator backend if transpile=="sim"
            if   self.transpile == "qc" or self.flag_real_qc:
                backend         = backend_hw
                
            elif self.transpile == "sim":
                backend         = sim
            
            # Transpile with optimization_level=0 to preserve structure (and avoid aggressive rewrites)
            self.qc_transpiled  = transpile(self.qc                     ,
                                            backend            = backend,
                                            optimization_level = 0      )

            # Optionally draw the transpiled circuit
            if self.print_circuit:
                fig         = self.qc_transpiled.draw(output = "mpl")
                plt.show()


            if self.flag_real_qc:
                
                # Use Qiskit Runtime Sampler to submit the circuit to real hardware
                sampler     = Sampler(mode = backend_hw)

                # Submit job with explicit shot count
                job         = sampler.run([self.qc_transpiled], shots = self.n_shots)  
                
                # Retrieve results and convert to counts
                result      = job.result()

                self.counts = result[0].join_data().get_counts()
                
            elif self.flag_real_qc == False:
                # Execute on the Aer simulator backend and extract counts
                job         = sim.run(self.qc_transpiled, shots = self.n_shots)
                self.counts = job.result().get_counts()



    def get_Histogram(self):
        # Convenience visualization: plot the measurement counts histogram (keys are bitstrings, values are counts)
        plt.bar(   x        = self.counts.keys(  ),
                   height   = self.counts.values())
        plt.xticks(rotation = 90)
        plt.show()


    def get_solution_classic(self, A, b):
        # Classical reference solution x = A^{-1} b (used for validation and sign inference)
        A_inv = np.linalg.inv(A)
        x_c   = np.dot(A_inv, b)
        return x_c


    def euclidean_distance(self, a, b):
        """
        Compute the Euclidean distance between two vectors a and b.

        Parameters
        ----------
        a, b : Sequence[float]
            Input vectors of the same length.

        Returns
        -------
        float
            Euclidean distance: sqrt(sum((a_i - b_i)**2 for i in range(len(a)))).
        """

        # Normalize both vectors before comparing (distance is computed between direction vectors)
        a = a / np.linalg.norm(x = a)
        b = b / np.linalg.norm(x = b)

        # Euclidean distance between the normalized vectors
        return np.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


    def get_solution_quantum(self, kwargs):

        # Extract amplitudes (up to a global normalization) from measurement probabilities:
        # counts are assumed normalized probabilities (or scaled later); take sqrt to get amplitude estimates
        q_probs      = np.array([abs(kwargs["counts"][c]) for c in kwargs["combination"]])**0.5

        # Impose signs based on the classical reference solution (resolves sign ambiguity from probability-only data)
        q_probs_sign = np.sign(kwargs["x_c"])
        q_probs      = q_probs * q_probs_sign

        # Reconstruct b_hat = A * q_probs (proxy for the RHS induced by the estimated quantum state)
        b_hat        = np.dot(kwargs["A"], q_probs)

        # Estimate per-component scaling λ from matching normalized RHS (heuristic rescaling step)
        lmbd         = np.abs(kwargs["b_norm"] / b_hat)

        # Construct quantum estimate x_q by scaling amplitude vector elementwise
        x_q          = q_probs*lmbd

        # Rescale back to the original (unnormalized) b using stored b_factor (A_factor is commented out)
        x_q          = (x_q*kwargs["b_factor"])

        # Round for stable reporting / storage
        x_q          = np.round(x_q, 4)

        # Compute corresponding reconstructed RHS b_q = A * x_q (in the same normalization convention as used above)
        b_q          = (kwargs["A"] @ x_q)
        b_q          = np.round(b_q, 4)

        # Distance metrics between classical and quantum solutions (directional comparison via normalization in euclidean_distance)
        x_norm2      = self.euclidean_distance(kwargs["x_c"], x_q)
        b_norm2      = self.euclidean_distance(kwargs["b"  ], b_q)

        # Componentwise relative error ratios and their means (used as summary accuracy measures)
        ratio_x      = abs(abs(x_q - kwargs["x_c"]) / kwargs["x_c"])
        ratio_x_mean = np.mean(ratio_x)
        ratio_b      = abs(abs(b_q - kwargs["b"  ]) / kwargs["b"  ])
        ratio_b_mean = np.mean(ratio_b)

        # Optional mode: return only a reduced metric (ratio_x_mean) when kwargs["flag"] is set
        if kwargs["flag"]:
            return np.array([ratio_x_mean])
        else:
            # Full return: reconstructed x and b, error ratios, mean ratios, and distance metrics
            return x_q, np.round(b_q, 4), ratio_x, ratio_x_mean, ratio_b, ratio_b_mean, x_norm2, b_norm2


    def get_best_solution(self, counts, A, A_factor, b, b_norm, b_factor, x_c):

        # Hard-coded subset of bitstrings corresponding to the "accepted" ancilla/postselection outcomes
        # (interpreted here as the set used to reconstruct the solution amplitudes)
        best        = ["0001", "0011", "0101", "0111", "1001", "1011", "1101", "1111"]

        # Effective number of shots in the accepted subspace (sum of counts over the chosen bitstrings)
        N_eff_real  = sum(counts[k] for k in best)

        # Normalize counts to empirical probabilities by dividing by total shots
        counts      = dict((k, v/self.n_shots) for k, v in counts.items())

        # Bundle parameters used by get_solution_quantum into a single dict for convenience
        kwargs_list = {"counts":      counts  ,
                       "combination": best    ,
                       "A":           A       ,
                       "A_factor":    A_factor,
                       "b":           b       ,
                       "b_norm":      b_norm  ,
                       "b_factor":    b_factor,
                       "x_c":         x_c     ,
                       "flag":        False   }

        # Reconstruct quantum solution and associated error metrics
        x_q, b_q, ratio_x, ratio_x_mean, ratio_b, ratio_b_mean, x_norm2, b_norm2 = self.get_solution_quantum(kwargs_list)

        # Optional diagnostic printing for debugging / paper figures
        if self.print_solution:

            print("\nKombination: ", list(best)  )
            print("x classic:     ", x_c         )
            print("x quantum:     ", x_q         )
            print("ratio x:       ", ratio_x     )
            print("ratio_mean x:  ", ratio_x_mean)
            print("b classic:     ", b           )
            print("b quantum:     ", b_q         )
            print("ratio b:       ", ratio_b     )
            print("ratio_mean b:  ", ratio_b_mean)
            print("x_norm2:       ", x_norm2     )
            print("b_norm2:       ", b_norm2     )

        # Return summary metrics used downstream: solution distance, RHS distance, and effective accepted shots
        return x_norm2, b_norm2, N_eff_real


    def get_solution(self):

        def help_1xA_2xA_1xb_1xb(counts, index):
            # Build a full dictionary over all basis states of the system register,
            # appending the given ancilla/index bit (e.g., "...1") and filling missing keys with 0.
            temp                 = dict()
            for i in range(2**self.nb):
                nr               = format(i, "0" + str(self.nb) + "b")

                try:
                    x1           = counts[nr + index]
                except:
                    x1           = 0

                temp[nr + index] = x1

            return temp
            
        # Select counts conditioned on the chosen ancilla/index outcome (here: index="1")
        counts1                  = help_1xA_2xA_1xb_1xb(counts = self.counts, index = "1")

        # Compute best-solution metrics from the filtered/conditioned count dictionary
        x_norm2, b_norm2, N_eff  = self.get_best_solution(counts1, self.A1, self.A1_factor, self.b1, self.b1_norm, self.b1_norm_factor, self.x_c1)
            
        # Return main scalar metrics used in experiments
        return x_norm2, b_norm2, N_eff


    def run(self):
        # Execute the full workflow: parameter setup, circuit build, execution, and count extraction
        self.get_counts()

        # For QS mode, return the constructed QuantumCircuit object for inspection / external handling if needed
        if self.package == "QS":
            return self.qc
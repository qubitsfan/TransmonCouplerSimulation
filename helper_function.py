import numpy as np
import qutip as qt
from sympy.physics.quantum import Dagger
from Floquet_perturbation_theory import *

### --- For Transmoon-Coupler-Transmon system Simplified Hamiltonian --- ###

def HardCore(expr, a1, a2, ac):
    """remove the off-diagonal terms in each of the single bosonnic modes,
    which corresponds to the hard-core boson limit
    where each mode can only have 0 or 1 excitation.
    This is a common approximation in the dispersive regime
    where the anharmonicity is large compared to the coupling strength,
    so that higher excitations are energetically suppressed.

    Args:
        expr (): sympy expression containing the bosonic ladder operators a1, a2, ac and their daggers

    Returns:
        _type_: sympy expression with the off-diagonal terms removed
    """
    expr = expr.replace(
        lambda x: x.is_Pow
        and x.exp >= 2
        and (
            x.base == a1
            or x.base == a2
            or x.base == ac
            or x.base == Dagger(a1)
            or x.base == Dagger(a2)
            or x.base == Dagger(ac)
        ),
        lambda x: 0,
    )
    return expr

### --- For Transmon-Coupler-Transmon system Diagonalization --- ###
def SortedFRFSpectrum(evals, evecs_qobj, dim_q1, dim_r, dim_q2, fidelity_threshold=0.5):
    """
    Sort eigenvalues/eigenvectors of a two-qubit–resonator system into branches
    |i_q1, n_r, i_q2> based on total qubit excitation number (i_q1 + i_q2)
    and resonator excitation n_r.

    The algorithm:
    1. Groups qubit states by total excitation N_q = i_q1 + i_q2.
    2. For each N_q branch, starts at n_r = 0 and finds eigenstates with
       maximal fidelity to the bare states |i_q1, 0, i_q2>.
    3. Then "climbs" the resonator ladder using the resonator creation operator
       to locate higher n_r levels.
    4. Returns fixed-size arrays for convenience.

    Parameters
    ----------
    evals : array_like
        Eigenvalues (1D array).
    evecs_qobj : list of Qobj
        Corresponding normalized eigenvectors.
    dim_q1, dim_q2, dim_r : int
        Dimensions of the two qubits and the resonator Hilbert space.
    fidelity_threshold : float, optional
        Minimal fidelity to consider a match valid.

    Returns
    -------
    sorted_evals : np.ndarray, shape (dim_q1, dim_r, dim_q2)
    sorted_evecs : np.ndarray, shape (dim_q1, dim_r, dim_q2), dtype=object
    """

    used = set()
    sorted_evals = np.full((dim_q1, dim_r, dim_q2), np.nan)
    sorted_evecs = np.empty((dim_q1, dim_r, dim_q2), dtype=object)

    # Resonator creation operator
    a_r = qt.destroy(dim_r)
    excit_op = qt.tensor(qt.qeye(dim_q1), a_r.dag(), qt.qeye(dim_q2))

    def find_best_match(target):
        """Find unused eigenstate with maximal fidelity to target."""
        best_idx, best_fid = None, -1.0
        for i, state in enumerate(evecs_qobj):
            if i in used:
                continue
            fid = qt.fidelity(state, target)
            if fid > best_fid:
                best_fid, best_idx = fid, i
        return best_idx, best_fid

    # Maximum total qubit excitation
    max_qubit_exc = (dim_q1 - 1) + (dim_q2 - 1)

    # Loop over total qubit excitation manifolds
    for Nq in range(max_qubit_exc + 1):

        # --- Step 1: find all (iq1, iq2) combos with this total excitation
        qubit_pairs = [
            (iq1, iq2)
            for iq1 in range(dim_q1)
            for iq2 in range(dim_q2)
            if iq1 + iq2 == Nq
        ]

        for iq1, iq2 in qubit_pairs:
            # --- Step 2: find best match for resonator ground state
            target = qt.tensor(
                qt.basis(dim_q1, iq1), qt.basis(dim_r, 0), qt.basis(dim_q2, iq2)
            )
            idx, fid = find_best_match(target)
            if idx is None or fid < fidelity_threshold:
                continue

            used.add(idx)
            sorted_evals[iq1, 0, iq2] = evals[idx]
            # keep phase continuity when possible
            if (evecs_qobj[idx].dag() @ target).real < 0:
                sorted_evecs[iq1, 0, iq2] = -evecs_qobj[idx]
            else:
                sorted_evecs[iq1, 0, iq2] = evecs_qobj[idx]
            current_state = evecs_qobj[idx]

            # --- Step 3: climb resonator ladder for this qubit pair
            for jr in range(1, dim_r):
                excited = (excit_op * current_state).unit()
                idx, fid = find_best_match(excited)
                if idx is None or fid < fidelity_threshold:
                    break
                used.add(idx)
                sorted_evals[iq1, jr, iq2] = evals[idx]
                # keep phase continuity when possible
                if (evecs_qobj[idx].dag() @ excited).real < 0:
                    sorted_evecs[iq1, jr, iq2] = -evecs_qobj[idx]
                else:
                    sorted_evecs[iq1, jr, iq2] = evecs_qobj[idx]
                current_state = evecs_qobj[idx]

    return sorted_evals, sorted_evecs

### --- 

def Psi_t_FloquetPerturb(rH, rW, wd, amp, resonances, E, V_posHarm, initial_state, t_list):
    """
    Calculate time-evolved wavefunction using Floquet perturbation theory.
    
    This function computes the time evolution of a quantum state under a periodically
    driven Hamiltonian using Floquet perturbation theory to a specified order. It 
    calculates both the effective Floquet Hamiltonian and the micromotion operator,
    then evolves the initial state and normalizes the result at each time point.

    Args:
        rH (int): Order of perturbation theory for the effective Hamiltonian.
        rW (int): Order of perturbation theory for the micromotion operator W.
        wd (float): Drive frequency (angular frequency of the periodic modulation).
        resonances (dict): Dictionary mapping state indices to their Floquet sectors,
            e.g., {a:na, b:nb, ...}. Such that E[a]-na*wd ~ E[b]-nb*wd.
        E (array_like): Energy eigenvalues of the undriven system (1D array).
        V_posHarm (array_like or list): Matrix elements of the positive harmonic
            drive coupling (2D array) or list of such matrices for multiple harmonics.
        initial_state (array_like): Initial state vector in the Hilbert space (1D array).
        t_list (array_like): Array of time points at which to evaluate the wavefunction.

    Returns:
        np.ndarray: Time-evolved wavefunction, shape (dim, len(t_list)), where dim is
            the dimension of the Hilbert space. The wavefunction is normalized at each
            time point. Each column psi_t[:, i] represents the state at time t_list[i].
    """
    
    t = sy.symbols("t", real=True)

    ## Substitution time in analytical solution and normalize wavefunction
    def substitue_time_normalize(psi_t_analyt):
        func = sy.lambdify(t, sy.Array(psi_t_analyt))
        psi_t_analyt_sub = np.array([func(tval) for tval in t_list]).T
        ## Normalization of wavefunction
        norm_t = np.sum(np.abs(psi_t_analyt_sub) ** 2, axis=0)
        psi_t_analyt_sub = np.einsum("kt,t->kt", psi_t_analyt_sub, 1 / np.sqrt(norm_t))
        return psi_t_analyt_sub
    
    Heff_matrix = Heff_Floquet_matrix_summed(
        rH, wd, resonances, E, amp / 2 * V_posHarm 
        )
    W_element = W_Floquet_elements(
        rW, wd, resonances, E, amp / 2 * V_posHarm,  
        )
    psi_t = substitue_time_normalize(
        dynamics_eff(
            initial_state, rH, rW,
            Heff_matrix, W_element,
            t, wd, resonances, E,
            )
        )
    return psi_t



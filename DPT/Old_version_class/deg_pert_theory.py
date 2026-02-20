"""
Code to compute the coefficients of degenerate perturbation theory.
Execute this file to save the coefficients up to order 10.
Change the range in the __main__ below to compute other orders.

Evaluates the recursive equation of the paper of Soliverez
    C E Soliverez (1969) J. Phys. C: Solid State Phys. 2 2161
    DOI 10.1088/0022-3719/2/12/301
The Hamiltonian is 
    H = H0 + V
We note P the projector on the degenerate subspace of H0 of energy E0.
The effective Hamiltonian in the subspace P is written in terms of P and the resolvant operator
    R = (1 - P)/(E0 - H).

The instances of the class OperatorString represent an order n operator of the form
    c S^{m_0} V S^{m_1} ... V S^{m_n}
where c is a scalar coefficient, and where the exponents m_0, ..., m_n are integers where
    S^0 = P   and    S^m = R^m  for m>0.
So the class OperatorString has two attributes:
    string: the list [m_0, ..., m_n] of exponents.
    coeff: the coefficient c.

The operators are linear combination of operator strings. 
The data of the class Operator is 
    terms: the list [string1, string2, ...] of strings of the class OperatorString.

The class OperatorString and Operator have methods corresponding to the main algebraic operations.

WARNING ON CONVENTION:
    Note that we have a difference of sign in our notations compared to [C E Soliverez (1969) J. Phys. C: Solid State Phys. 2 2161].
    We use R = (1 - P)/(E0 - H) instead of the K = (1 - P)/(H - E0) = -R of Soliverez.
    As a result, we have
        [m_0, ..., m_n]_{here} = (-1)^{m_0 + ... + m_n} [m_0, ..., m_n]_{Soliverez}.
    This enables to remove the alternated sign in the simple term [1, ..., 1] appearing at every order in the effective Hamiltonian.

The function Heff(n) returns the effective Hamiltonian at order n, in the format Operator.

print(Heff(n)) print all the operator strings and associated coefficient of the effective Hamiltonian of order n.

The function Save_Heff(n) saves the coefficients of order n of the effective Hamiltonian in a txt file 'DPT_coefficients_order_{n}.txt' in the format <list of exponents [m_{r-1}, ..., m_{1}]> <coeff c> (where the first and last exponents 0 for P are omitted). Such that
    Heff^{(n)} = sum_{m_1+...+m_{n-1}=n-1} c_{m_1, ..., m_{n-1}} P V S^{m_1} ... V S^{m_{n-1}} V P

The function Save_W(n) saves the coefficients of order n of the unitary W of DPT in a txt file 'DPT_unitaryW_coefficients_order_{n}.txt' in the format <list of exponents [m_r, ..., m_{1}]> <coeff c> (where the last exponents 0 for P is omitted). Such that
    V^{(n)} = sum_{m_1+...+m_{n-1}=n-1} c_{m_0, ..., m_{n-1}} S^{m_0} V S^{m_1} ... V S^{m_{n-1}} V P
"""

from copy import deepcopy
import numpy as np

class OperatorString:
    """
    Operator string of the form [m_0, ..., m_n] and its associated coefficient.
    self.string is the list [m_0, ..., m_n] of exponents of the string
    self.coeff is the coefficient.
    """
    def __init__(self, string=[], coeff=0):
        self.string = string
        self.coeff = coeff

    def __str__(self) -> str:
        return f"String: {self.string}   Coeff: {self.coeff}"
    
    def __rmul__(self, scalar):
        """
        Multiply the contribution by a scalar (scalar on the left).
        """
        result = OperatorString()
        result.string = deepcopy(self.string)
        result.coeff = scalar*self.coeff
        return result

    def dot(self, other):
        """
        Product between two operator strings.
        """
        if (self.string[-1] == 0) ^ (other.string[0] == 0):
                ## If the last operator of first string is P and the first operator of second string is not P, or vice-versa, then return 0 because product P*R=R*P=0.
            return OperatorString([], 0)
        
        result = OperatorString()
        result.coeff = self.coeff * other.coeff
        result.string = [*self.string[:-1], self.string[-1] + other.string[0], *other.string[1:]]
        return result
    
    def __mul__(self, other):
        return self.dot(other)
    
    def Vdot(self, other):
        """
        First string times V times second string
        """
        result = OperatorString()
        result.coeff = self.coeff * other.coeff
        result.string = [*self.string, *other.string]
        return result
    
    def dag(self):
        """
        Hermitian conjugated of operator string
        """
        return OperatorString(list(reversed(self.string)), self.coeff)

class Operator:
    """
    List of operator string (and there coefficients) representing an operator as a sum of these string with corresponding coefficients.
    self.terms is the list [string1, string2, ...] of all terms of the operator, where each string is an instance of OperatorString
    """
    def __init__(self, term=None):
        if term is None:
            self.terms = []
        elif isinstance(term, OperatorString):
            self.terms = [term]
        # else:
        #     ## If give already a list of OperatorString
        #     self.terms = deepcopy(terms)

    def __str__(self) -> str:
        return "Terms:\n\t"+"\n\t".join([string.__str__() for string in self.terms])
    
    def __rmul__(self, scalar):
        """
        Multiply all the terms by a scalar (scalar on the left).
        """
        result = Operator()
        result.terms = deepcopy(self.terms)
        for i in range(len(result.terms)):
            result.terms[i] = scalar*result.terms[i]
        return result

    def __add__(self, other):
        """
        Sum of two operators
        """
        result = Operator()
        result.terms = deepcopy(self.terms)
        if isinstance(other, OperatorString):
            ### Case where we add a string
            if other.coeff == 0:
                ### Case where we add 0. We do nothing
                return result
            for i in range(len(result.terms)):
                if result.terms[i].string == other.string:
                    ### Add the coefficient if the string is already in the operator
                    result.terms[i].coeff += other.coeff
                    ### Remove the term if now the coeff is zero
                    if np.isclose(result.terms[i].coeff, 0):
                        del result.terms[i]
                    return result
            ### If the string was not in the operator, we add it
            result.terms.append(other)
            return result
        else:
            ## Case where we add another operator: add all the string of the other operator, using the above implementation.
            for term in other.terms:
                result = result + term
            return result
        
    def __sub__(self, other):
        """
        Substraction of two operators
        """
        return self + -1*other

    def dot(self, other):
        """
        Multiplication of to operators. Sum of product of all strings of the two operators.
        """
        result = Operator()
        for s1 in self.terms:
            for s2 in other.terms:
                result += s1*s2
        return result
    
    def __mul__(self, other):
        return self.dot(other)
    
    def Vdot(self, other):
        """
        First operator times V times second operator
        """
        result = Operator()
        for s1 in self.terms:
            for s2 in other.terms:
                result += s1.Vdot(s2)
        return result

    def dag(self):
        """
        Hermitian conjugate of the operator (term by term)
        """
        result = Operator()
        for term in self.terms:
            result += term.dag()
        return result
    
    def get_coeff(self, string):
        """
        (for tests)
        Return the coefficient of a given string if it is part of the operator.
        """
        for term in self.terms:
            if term.string == string:
                return term.coeff
        return 0
    
    def non_trivial_terms(self):
        """
        (for tests)
        Return the terms of coeff different from 1, 0.5, -0.5
        """
        result = Operator()
        for term in self.terms:
            if term.coeff not in [1, 0.5, -0.5]:
                result.terms.append(term)
        return result
    
    def trivial_terms(self):
        """
        (for tests)
        Return the terms of coeff 1, 0.5, or -0.5
        """
        return self - self.non_trivial_terms()

def L(n):
    """
    Order n of the operator L of Soliverez defined recursively.
    Output: type Operator
    """
    P = Operator(OperatorString([0], 1))
    R = Operator(OperatorString([1], 1))
    if n == 0:
        return P
    elif n == 1:
        return R.Vdot(P)
    else:
        result = R.Vdot(L(n-1))
        for l in range(1, n):
            result = result - R*L(l).Vdot(L(n-l-1))
        return result
    
def N(n):
    """
    Order n of the operator N of Soliverez defined from L.
    Output: type Operator
    """
    result = Operator()
    for l in range(0, n+1):
        result += L(l).dag()*L(n-l)
    return result

def N_posSqrRoot(n):
    """
    Order n of the operator N^{1/2} of Soliverez defined recursively and from N.
    Output: type Operator
    """
    P = Operator(OperatorString([0], 1))
    if n == 0:
        return P
    elif n == 1:
        ## Order 1 is 0
        return Operator()
    else:
        result = 1/2*N(n)
        for l in range(1, n):
            result += -1/2*N_posSqrRoot(l)*N_posSqrRoot(n-l)
        return result
    
def N_negSqrRoot(n):
    """
    Order n of the operator N^{-1/2} of Soliverez defined recursively and from N^{1/2}
    Output: type Operator
    """
    P = Operator(OperatorString([0], 1))
    if n == 0:
        return P
    else:
        result = Operator()
        for l in range(n):
            result -= N_negSqrRoot(l)*N_posSqrRoot(n-l)
        return result
    
def Heff(n):
    """
    Order n of the effective Hamiltonian defined from N^{-1/2}, L, and N^{1/2}
    Output: type Operator
    """
    result = Operator()
    for l in range(n):
        for m in range(n-l):
            result += N_posSqrRoot(l).Vdot(L(m))*N_negSqrRoot(n-l-m-1)
    return result

def W(n):
    """
    Operator W of Soliverez associating the eigenstate of the total Hamiltonian |A> from the eigenstate |A'> of the effective Hamiltonian: 
        If    Heff |A'> = Delta |A'>
        Then  H|A> = (E0 + Delta)|A>
        where |A> = W|A'>.
    Because |A'> is only the component of |A> in the degenerate subspace.
    """
    result = Operator()
    for l in range(0, n+1):
        result += L(l)*N_negSqrRoot(n-l)
    return result


def Save_Heff(n):
    """
    Save coefficients of order n of the effective Hamiltonian in a txt file.
    Save it in 'DPT_coefficients_order_{n}.txt'
    """
    H = Heff(n)
    ## Data gathered in a list of list [ [m2, ..., mk, coeff], ...] for sorting. Ignore the first and last exponents which are necesserally 0.
    data = [[*term.string[1:-1], term.coeff] for term in H.terms]

    with open(f'DPT_coefficients_order_{n}.txt', 'w') as f:
        f.write(f"# Coefficient of the effective Hamiltonian in degenerated perturbation theory at order r={n}\n")
        f.write("# Format: <list of exponents [m_{r-1}, ..., m_{1}]> <coeff>\n")
        for m_and_coeff in sorted(data):
            f.write('\t'.join(str(mj) for mj in m_and_coeff[:-1])+'\t'+str(m_and_coeff[-1])+'\n')

def Save_W(n):
    """
    Save coefficients of order n of the unitary W in a txt file.
    Save it in 'DPT_unitaryW_coefficients_order_{n}.txt'
    """
    ## Data gathered in a list of list [ [m2, ..., mk, coeff], ...] for sorting. Ignore the last exponent which is necesserally 0.
    data = [[*term.string[:-1], term.coeff] for term in W(n).terms]

    with open(f'DPT_unitaryW_coefficients_order_{n}.txt', 'w') as f:
        f.write(f"# Coefficient of the unitary W of degenerated perturbation theory at order r={n}\n")
        f.write("# Format: <list of exponents [m_r, ..., m_{1}]> <coeff>\n")
        for m_and_coeff in sorted(data):
            f.write('\t'.join(str(mj) for mj in m_and_coeff[:-1])+'\t'+str(m_and_coeff[-1])+'\n')


if __name__ == "__main__":
    """
    Approximate computation time:
    Order 7: 3s
    Order 8: 1min
    Order 9: 15min
    Order 10: 4h35
    """
    import time
    ### Compute and save the coefficients up to order 10. Change the range to compute different orders.
    Save_W(1)
    for n in range(2, 11):
        T = time.time()
        Save_Heff(n)
        Save_W(n)
        print(f"Order {n} finished in {time.time() - T} s")
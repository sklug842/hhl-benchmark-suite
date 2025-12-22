# SciPy: generate random orthogonal matrices (Haar-distributed) used to construct symmetric matrices with a prescribed spectrum
from   scipy.stats import ortho_group

# NumPy: numerical arrays and linear algebra utilities used throughout the matrix/vector generation
import numpy       as     np


class get_A_x_b():
    def __init__(self):
        pass  

    def get_symmetric_matrix(self, ev):

        # Draw a random orthogonal matrix Q (Haar-distributed) of dimension len(ev)
        orthogonal_matrix = ortho_group.rvs(dim=len(ev))

        # Construct a symmetric matrix A = Q diag(ev) Q^T with prescribed eigenvalues ev
        return orthogonal_matrix @ np.diag(ev) @ orthogonal_matrix.T
    

    def get_pairwise_diffs(self, ev):

        # Build a list of left/right nearest-neighbor gaps for each eigenvalue
        # (endpoints use a placeholder gap of 1 on the missing side)
        pairwise_diffs = []
        for i in range(len(ev)):
            left_diff  = 1
            right_diff = 1
            if i > 0:
                left_diff  = ev[i    ] - ev[i - 1]
            if i < len(ev) - 1:
                right_diff = ev[i + 1] - ev[i    ]

            # Append both adjacent gaps for later min-gap checks
            pairwise_diffs.append(left_diff )
            pairwise_diffs.append(right_diff)
        
        return pairwise_diffs


    def get_eigenval(self, n_i, ratio, flag_neg):

        # Sample a target spectral window [low, high] with high capped below 1
        # and enforce the desired condition number via low = high/ratio
        high = np.random.uniform(min(0.3*ratio, 0.95), 0.95)
        low  = high/ratio

        # Loop control: keep sampling until spacing constraints are satisfied
        flag = True
        
        # Minimum spacing constraint depends on whether negative eigenvalues may appear
        if flag_neg:
            min_diff = 0.04
        else:
            min_diff = 0.02
        
        while flag:

            # Sample interior eigenvalues uniformly within (low, high), keep 2-decimal grid,
            # and then pin endpoints exactly to low and high to realize the target ratio
            ev_1          = np.sort(np.round(a       = np.random.uniform(low  = low +min_diff ,
                                                                         high = high-min_diff ,
                                                                         size = n_i          ),
                                            decimals = 2                                     ))
            ev_1[ 0]      = low
            ev_1[-1]      = high

            if flag_neg:
                
                # Choose a fixed-size random subset of indices that will later be negated
                n         = 4
                
                subset    = np.sort(np.random.choice(a       = np.arange(len(ev_1)),
                                                     size    = n                   ,
                                                     replace = False              ))

                # Build two temporary spectra used only for spacing tests:
                # - ev_1_temp: transformed version of the would-be negative subset
                # - ev_2_temp: transformed version of the remaining positive subset
                # The /2 scaling and modular wrap keep values in [0,1) for these spacing heuristics
                ev_1_temp = np.sort((            -ev_1[ subset] / 2) % 1)
                ev_2_temp = np.sort(ev_1[np.setdiff1d(np.arange(len(ev_1)), subset)] / 2)
                
            else:
                # If no negatives, use the same spectrum for both spacing checks
                ev_1_temp = ev_1
                ev_2_temp = ev_1

            # Accept the sample only if:
            # (i) all nearest-neighbor gaps in ev_1_temp are >= min_diff
            # (ii) all nearest-neighbor gaps in ev_2_temp are >= min_diff
            # (iii) the smallest of ev_1_temp is sufficiently separated from the largest of ev_2_temp
            if (np.sum( np.abs( np.array(self.get_pairwise_diffs(ev_1_temp))) < min_diff) == 0 and
                np.sum( np.abs( np.array(self.get_pairwise_diffs(ev_2_temp))) < min_diff) == 0 and
                abs(ev_1_temp[0]-ev_2_temp[-1]) > min_diff):
                flag = False

        # After acceptance, actually flip the selected eigenvalues to negative (if enabled)
        if flag_neg:
            ev_1[subset] = -ev_1[subset]

        # Return eigenvalues (sorted up to the sign flips applied above)
        return ev_1



    def normalized_with_min_amplitude(self, n: int, m: float) -> np.ndarray:
        """
        Return a length-n vector b such that

        1. \sum_i b_i^2 = 1
        2. \min_i b_i = m
        3. b_i \ge m for all i

        Requires: 0 \le m \le 1 / \sqrt{n}.
        """

        # Feasibility check: if m is too large, no unit-norm nonnegative vector can have all entries >= m
        if not (0 <= m <= 1/np.sqrt(n)):
            raise ValueError(f"m must satisfy 0 <= m <= 1/sqrt({n})")

        # Choose one coordinate to be the unique minimum (exactly m)
        min_idx       = np.random.randint(n)

        # Remaining squared amplitudes must sum to 1 - m^2 to keep ||b||_2 = 1
        remainder     = 1.0 - m**2

        # Randomly distribute the remaining squared mass over the other n-1 coordinates
        # using a Dirichlet distribution (i.e., a random simplex point)
        weights       = np.random.dirichlet(np.ones(n-1))

        # Assemble squared amplitudes: one fixed at m^2, others share the remainder
        s             = np.empty(n)
        s[min_idx]    = m**2
        other_idxs    = [i for i in range(n) if i != min_idx]
        s[other_idxs] = remainder * weights

        # Convert squared amplitudes to amplitudes (nonnegative)
        b             = np.sqrt(s)

        # Return b; caller may introduce random signs if desired
        return b


    def get_x_b(self, A, threshold_x, threshold_b):

        # Rejection sampling loop: keep drawing b until both b_norm and x_norm meet tight threshold criteria
        flag        = False

        while flag == False:

            # Draw a unit vector b_norm with minimum absolute amplitude threshold_b,
            # then assign random signs elementwise to allow both positive and negative entries
            b_norm  = self.normalized_with_min_amplitude(n = len(A)     ,
                                                         m = threshold_b) * np.random.choice([-1, 1], size=len(A))
            
            # Scale b_norm to obtain an unnormalized right-hand side b (random magnitude in [2,5])
            b       = b_norm * np.random.uniform(2,5)

            # Solve A x = b; using solve is numerically preferable to explicit inversion
            x       = np.linalg.solve(A, b)

            # Normalize solution vector to unit norm
            x_norm  = x / np.linalg.norm(x = x  )

            # Accept only if the minimum absolute entry in x_norm and b_norm is within 1% of the target threshold
            # (i.e., close to the boundary, not merely above it)
            if  (np.abs((np.min(np.abs(x_norm)) - threshold_x) / threshold_x) <= 0.01) and \
                (np.abs((np.min(np.abs(b_norm)) - threshold_b) / threshold_b) <= 0.01):
               
                flag = True

        # Return solution and RHS in both raw and normalized forms
        return x, x_norm, b, b_norm



    def get_A_x_b(self, n_i, cond_numbr, flag_neg, threshold_x, threshold_b):
    
        # Generate eigenvalues with the requested condition-number ratio and sign structure
        ev        = self.get_eigenval(n_i      = n_i       ,
                                      ratio    = cond_numbr,
                                      flag_neg = flag_neg  )

        # Construct symmetric matrix A with spectrum ev
        A         = self.get_symmetric_matrix(ev = ev      )

        # Rejection sampling loop (redundant with checks inside get_x_b, but enforces the same criteria here)
        flag      = False

        while flag == False:

            # Sample (x,b) consistent with A and with boundary-tight threshold constraints
            x, x_norm, b, b_norm = self.get_x_b(A           = A          ,
                                                threshold_x = threshold_x,
                                                threshold_b = threshold_b)
            
            # Accept only if the tight 1% boundary condition holds for both x_norm and b_norm
            if (np.abs((np.min(np.abs(x_norm)) - threshold_x) / threshold_x) <= 0.01) and \
               (np.abs((np.min(np.abs(b_norm)) - threshold_b) / threshold_b) <= 0.01):
                flag = True
        
        # Return the full tuple used downstream: matrix, solution, normalized solution, rhs, normalized rhs
        return A, x, x_norm, b, b_norm
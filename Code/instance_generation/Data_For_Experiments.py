# Standard library imports for timing the experiment / run duration
from   datetime      import datetime

# Numerical computing library (NumPy) for arrays, dtypes, and linear-algebra-related utilities
import numpy         as     np

# Import helper class used for eigenvalue and matrix construction
from   HHL_get_A_x_b import get_A_x_b

# Print NumPy floating-point numbers with a fixed precision for readability in logs/debug
np.set_printoptions(precision=4)

# Output directory (intended target for saving the generated record array)
path = "../../Data/"

# Number of repeated fields per suffix (e.g., N_eff_<suffix>_1..K, norms, etc.)
K    = 10

def rep(prefix, t, n):
    """('prefix_1'..'prefix_n', t)"""
    # Convenience helper to create structured-array fields with numbered suffixes
    return [(f"{prefix}_{i}", t) for i in range(1, n + 1)]


# Base structured dtype fields:
# - Scalars: index, condition number, flags, expected effective shots, etc.
# - Objects: A, x, b and their normalized versions are stored as objects to accommodate array-like content
dtype_list = [("index"           , np.int64  ),
              ("cond_numbr"      , np.float64),
              ("flag_neg"        , np.int64  ),
              ("A"               , object    ),
              ("n_shots"         , np.float64),
              ("N_eff_expected"  , np.float64),
              ("threshold"       , np.float64),
              ("factor_threshold", np.float64),
              ("x"               , object    ),
              ("x_norm"          , object    ),
              ("b"               , object    ),
              ("b_norm"          , object    )]


# Extend dtype_list with per-method/per-backend statistics for multiple suffix variants.
# For each suffix:
# - K samples of N_eff, x_norm2, b_norm2 (as individual fields)
# - mean and std for each of those metrics
for suffix in ["PL_ideal"       ,
               "QS_ideal"       ,
               "QS_Aachen_ideal",
               "QS_Aachen_sim"  ,
               "QS_Aachen_qc"   ]:
    
    dtype_list += rep("N_eff_"        + suffix, np.int64  , K)                                         \
               + [(   "N_eff_mean_"   + suffix, np.float64   ), ("N_eff_std_"   + suffix, np.float64)] \
               + rep( "x_norm2_"      + suffix, np.float64, K)                                         \
               + [(   "x_norm2_mean_" + suffix, np.float64   ), ("x_norm2_std_" + suffix, np.float64)] \
               + rep( "b_norm2_"      + suffix, np.float64, K)                                         \
               + [(   "b_norm2_mean_" + suffix, np.float64   ), ("b_norm2_std_" + suffix, np.float64)]


# Final NumPy dtype for the structured records array
dtype = np.dtype(dtype_list)


def get_x_b_help(A, threshold_x, threshold_b):

    # Local import to keep dependency localized (and to avoid import costs if unused elsewhere)
    from HHL_get_A_x_b import get_A_x_b

    # Instantiate the helper/provider class that can generate x,b for a given A under thresholds
    get_A_x_b = get_A_x_b()

    # Loop until the generated normalized components meet the minimum thresholds
    flag      = False

    while flag == False:

        # Attempt to generate x and b (and their normalized versions) for the given matrix A
        x11, x11_norm, b1, b1_norm = get_A_x_b.get_x_b(A           = A          ,
                                                       threshold_x = threshold_x,
                                                       threshold_b = threshold_b)

        # Accept only if all absolute normalized entries are above the requested thresholds
        if (np.min(np.abs(x11_norm)) >= threshold_x and np.min(np.abs(b1_norm)) >= threshold_b):
            flag = True
    
    # Return the accepted sample (including A for convenience/consistency with call site)
    return A, x11, x11_norm, b1, b1_norm


# Timestamp for the beginning of the full generation run
start = datetime.now()

# Experiment sweep parameters:
# - conditional_numbers: condition-number ratios used to shape A
# - flags_neg: whether to include negative eigenvalues (or similar sign convention)
# - factors_threshold: multiplicative factors applied to the base threshold
conditional_numbers = [2, 3, 4, 5              ]
flags_neg           = [0, 1                    ]
factors_threshold   = [1, 2, 3, 4, 5           ]

# Fixed experiment parameters:
# - n_shots: baseline shot budget used to define N_eff and threshold
# - n_examples: number of A instances per condition-number and sign-flag
# - n_i: problem size parameter passed into eigenvalue generation
n_shots             = 10000
n_examples          = 10
n_i                 = 8

# Output filename encodes the sweep ranges and run parameters for traceability
name = "records_CN_" + str(conditional_numbers[0]) + "_" + str(conditional_numbers[-1]) + \
              "_FN_" + str(flags_neg[0]          ) + "_" + str(flags_neg[-1]          ) + \
              "_FT_" + str(factors_threshold[0]  ) + "_" + str(factors_threshold[-1]  ) + \
              "_NS_" + str(n_shots               )                                      + \
              "_NE_" + str(n_examples            )                                      + \
              "_ni_" + str(n_i                   ) + "_" + datetime.now().strftime("%Y%m%d_%H%M") + ".npy"

# Total number of records = all parameter combinations times n_examples
N                  = len(conditional_numbers) * len(flags_neg) * len(factors_threshold) * n_examples

# Preallocate N records as a structured NumPy array (for speed and fixed schema)
records            = np.zeros(N, dtype=dtype)

# Running index into the flat records array
i                  = 0

# Instantiate helper once for repeated use in the sweep
helper = get_A_x_b()

# Outer sweep over condition numbers (ratio parameter)
for cond_numbr in conditional_numbers:

    # Expected effective shots (problem-specific scaling), and base threshold derived from it
    N_eff     = n_shots / cond_numbr**2
    threshold = 1/np.sqrt(N_eff)

    # Cache A instances per flag_neg for this condition number so they can be reused across thresholds
    A_FN0 = []
    A_FN1 = []

    # Generate n_examples matrices A for each sign-flag variant
    for index in range(n_examples):

        # Generate eigenvalues for the symmetric matrix construction (flag_neg=0)
        ev_FN0 = helper.get_eigenval(n_i      = n_i       ,
                                     ratio    = cond_numbr,
                                     flag_neg = 0         )
        
        # Generate eigenvalues for the symmetric matrix construction (flag_neg=1)
        ev_FN1 = helper.get_eigenval(n_i      = n_i       ,
                                     ratio    = cond_numbr,
                                     flag_neg = 1         )

        # Construct and store symmetric matrices for both sign-flag settings
        A_FN0.append( helper.get_symmetric_matrix(ev = ev_FN0))
        A_FN1.append( helper.get_symmetric_matrix(ev = ev_FN1))

    # Sweep over sign-flag options, selecting the corresponding cached matrices
    for flag_neg in flags_neg:

        if   flag_neg == 0:
            A_temp = A_FN0

        elif flag_neg == 1:
            A_temp = A_FN1
        
        # Sweep over threshold scaling factors
        for factor_threshold in factors_threshold:

            # Progress logging for the current sweep point
            print("CN: ", cond_numbr, "  FN: ", flag_neg, "  FT: ", factor_threshold)

            # Per-run thresholds for x and b (scaled from the base threshold)
            threshold_x = threshold*factor_threshold
            threshold_b = threshold*factor_threshold

            # Generate x,b for each cached A instance under the given thresholds
            for index in range(n_examples):

                # Inner acceptance loop: keep sampling until x_norm and b_norm match thresholds tightly
                flag        = False

                while flag == False:

                    # Generate candidate x and b meeting the basic minimum-threshold constraints
                    A, x, x_norm, b, b_norm = get_x_b_help(A           = A_temp[index],
                                                           threshold_x = threshold_x  ,
                                                           threshold_b = threshold_b  )

                    # Additional acceptance criterion:
                    # enforce that the minimum absolute normalized entry is within 1% of the threshold
                    # (i.e., the sample is close to the boundary rather than far above it)
                    if (np.abs((np.min(np.abs(x_norm)) - threshold_x) / threshold_x) <= 0.01) and \
                       (np.abs((np.min(np.abs(b_norm)) - threshold_b) / threshold_b) <= 0.01):
                        flag = True

                # Fill record i with metadata and generated data
                records["index"           ][i] = index
                records["cond_numbr"      ][i] = cond_numbr
                records["flag_neg"        ][i] = flag_neg
                records["A"               ][i] = A
                records["n_shots"         ][i] = n_shots
                records["N_eff_expected"  ][i] = N_eff
                records["threshold"       ][i] = threshold
                records["factor_threshold"][i] = factor_threshold
                records["x"               ][i] = x
                records["x_norm"          ][i] = x_norm
                records["b"               ][i] = b
                records["b_norm"          ][i] = b_norm

                # Advance flat record index
                i += 1


# Save as a single .npy file:
print("save")
np.save(path + name, records)
print("saved")

# Timestamp for end of run and duration reporting
end = datetime.now()

print(f" ")
print(f"Start:    {start    }")
print(f"End:      {end      }")
print(f"Duration: {end-start}")
print(f" ")
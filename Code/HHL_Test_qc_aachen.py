# Wrapper around HHL execution used to run QS/PL variants and extract solution metrics
from   help_classes.wrapper_hhl import wrapper_HHL

# NumPy for loading/saving structured arrays and computing summary statistics (mean/std)
import numpy                    as     np

# Datetime for run timing and timestamping
from   datetime                 import datetime

# Qiskit Runtime service for IBM Quantum hardware execution (authentication + backend access)
from   qiskit_ibm_runtime       import QiskitRuntimeService

# sys for reading command-line arguments (cond_numbr_global, flag_neg)
import sys

# Print NumPy floating-point numbers with fixed precision for readable logs
np.set_printoptions(precision=4)

# Read selected condition number and spectrum class from CLI:
# - sys.argv[1]: condition number (int; one of 2, 3, 4, 5)
# - sys.argv[2]: negative-eigenvalue flag (int; 0 = positive-definite, 1 = indefinite / mixed-sign)
cond_numbr_global = int(sys.argv[1])
flag_neg          = int(sys.argv[2])

# Example execution (condition number = 2, flag_neg = 0):
# python3 HHL_Test_qc_aachen.py 2 0


#####
# Get Records
#####

# Base path where experiment records (.npy) are stored
path             = "../Data/"

# Select which precomputed record file to load based on the requested condition number
if   cond_numbr_global == 2:
    if   flag_neg == 0:
        records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520_CN_2.0_FN_0_20250918_114402_qc_min"
    elif flag_neg == 1:
        records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520_CN_2.0_FN_1_20250918_192906_qc_min"

elif cond_numbr_global == 3:
    if   flag_neg == 0:
        records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520_CN_3.0_FN_0_20250918_132207_qc_min"
    elif flag_neg == 1:
        records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520_CN_3.0_FN_1_20250918_194100_qc_min"

elif cond_numbr_global == 4:
    if   flag_neg == 0:
        records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520_CN_4.0_FN_0_20250918_115954_qc_min"
    elif flag_neg == 1:
        records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520_CN_4.0_FN_1_20250920_000208_qc_min"

elif cond_numbr_global == 5:
    if   flag_neg == 0:
        records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520_CN_5.0_FN_0_20250918_053526_qc_min"
    elif flag_neg == 1:
        records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520_CN_5.0_FN_1_20250918_084838_qc_min"



# Load records (structured NumPy array); allow_pickle=True because dtype contains object fields (e.g., A, b)
records      = np.load(file         = path + records_name + ".npy",
                       allow_pickle = True                        )



#####
# Print
#####

# Print run metadata for traceability in logs
print("")
print("records name:       ", records_name     )
print("")
print("Conditional Number: ", cond_numbr_global)
print("")
print("Spectrum flag:      ", flag_neg         )
print("")

#####
# start
#####

# Timestamp run start
start        = datetime.now()

# Iterate over all records in the loaded file
for i in range(len(records)):

    # Print per-record metadata to track progress and current configuration
    print("")
    print("Conditional Number: ", records["cond_numbr"      ][i])
    print("")
    print("Factor Threshold:   ", records["factor_threshold"][i])
    print("")

    # Accumulators for K=10 repeats on real hardware (QS Aachen qc)
    x_norm2_QS_Aachen_qc_array = []
    b_norm2_QS_Aachen_qc_array = []
    N_eff_QS_Aachen_qc_array   = []


    # Repeat K=10 executions to populate per-run fields and then compute mean/std
    for x in range(1,11):

        # Initialize IBM Quantum Runtime service (credentials + instance)
        # Note: tokens should not be hardcoded in submitted/public code; prefer env vars or secure config.
        service = QiskitRuntimeService(channel  = 'ibm_cloud', 
                                       instance = "123",
                                       token    = '123')


        #####
        # QS Aachen qc
        #####

        # Execute QS-based HHL targeting IBM Aachen with qc transpilation and real-hardware execution enabled
        wrapper_1xA_1xb_QS_Aachen_qc = wrapper_HHL(package        = "QS"                      ,
                                                   service        = service                   ,
                                                   backend        = "ibm_aachen"              ,
                                                   transpile      = "qc"                      ,
                                                   flag_real_qc   = 1                         ,
                                                   modus          = "1xA_1xb"                 ,
                                                   A1             = records["A"          ][i] ,
                                                   b1             = records["b"          ][i] ,
                                                   n_shots        = int(records["n_shots"][i]),
                                                   print_circuit  = 0                         ,
                                                   print_solution = 1                         ,
                                                   factor_t       = 0.99                      )

        # Run the job (circuit build/transpile/execute/postprocess handled inside wrapper)
        wrapper_1xA_1xb_QS_Aachen_qc.run()

        # Extract solution metrics for this repetition
        x_norm2_QS_Aachen_qc, b_norm2_QS_Aachen_qc, N_eff_QS_Aachen_qc = wrapper_1xA_1xb_QS_Aachen_qc.get_solution()

        # Store metrics for summary statistics
        x_norm2_QS_Aachen_qc_array.append(x_norm2_QS_Aachen_qc)
        b_norm2_QS_Aachen_qc_array.append(b_norm2_QS_Aachen_qc)
        N_eff_QS_Aachen_qc_array.append(  N_eff_QS_Aachen_qc  )

        # Write per-repeat metrics into structured record fields (preallocated upstream)
        records["x_norm2_QS_Aachen_qc_" + str(x)][i] = x_norm2_QS_Aachen_qc
        records["b_norm2_QS_Aachen_qc_" + str(x)][i] = b_norm2_QS_Aachen_qc
        records["N_eff_QS_Aachen_qc_"   + str(x)][i] = N_eff_QS_Aachen_qc


    # Compute and store mean/std across the K repeats for this record
    records["N_eff_mean_QS_Aachen_qc"  ][i] = np.mean(  N_eff_QS_Aachen_qc_array)
    records["N_eff_std_QS_Aachen_qc"   ][i] = np.std(   N_eff_QS_Aachen_qc_array)
    records["x_norm2_mean_QS_Aachen_qc"][i] = np.mean(x_norm2_QS_Aachen_qc_array)
    records["x_norm2_std_QS_Aachen_qc" ][i] = np.std( x_norm2_QS_Aachen_qc_array)
    records["b_norm2_mean_QS_Aachen_qc"][i] = np.mean(b_norm2_QS_Aachen_qc_array)
    records["b_norm2_std_QS_Aachen_qc" ][i] = np.std( b_norm2_QS_Aachen_qc_array)



# Timestamp run end
end = datetime.now()

####
# Save records
####

# Persist the updated records array back to the same filename (in-place update of stored metrics)
np.save(path + records_name + ".npy", records)


####
# Print
####

# Print wall-clock timing information for the run
print(f"                                   ")
print(f"Start:                  {start    }")
print(f"End:                    {end      }")
print(f"Duration:               {end-start}")
print(f"-----------------------------------")
print(f"                                   ")

# Echo configuration at end of run for log readability
print("")
print("Records Name:       ", records_name     )
print("")
print("Conditional Number: ", cond_numbr_global)
print("")
print("Spectrum flag:      ", flag_neg         )
print("")

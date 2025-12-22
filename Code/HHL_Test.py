# Wrapper around HHL execution (local project module) used to run different backends/simulators and extract solution metrics
from   help_classes.wrapper_hhl import wrapper_HHL

# NumPy for loading/slicing structured arrays and computing summary statistics (mean/std)
import numpy                    as     np

# Datetime for run timing and timestamping output filenames
from   datetime                 import datetime

# Qiskit Runtime service for IBM Quantum backends (authentication + backend access)
from   qiskit_ibm_runtime       import QiskitRuntimeService

# sys for reading command-line arguments (cond_numbr_global, flag_neg_global)
import sys

# Print NumPy floating-point numbers with fixed precision for readable logs
np.set_printoptions(precision=4)

# Read experiment selectors from command-line arguments (CLI):
# - sys.argv[1]: condition number (float) used to filter/select records for this run
# - sys.argv[2]: negative-eigenvalue flag (int; 0 = nonnegative spectrum, 1 = includes negative eigenvalues)
# - sys.argv[3]: GPU usage flag (int; 0 = force CPU execution, 1 = request GPU execution when supported)
cond_numbr_global = float(sys.argv[1])
flag_neg_global   = int(  sys.argv[2])
flag_use_gpu      = int(  sys.argv[3])


# Example execution (condition number = 2, flag_neg = 0, request GPU = 1):
# python3 HHL_Test.py 2 0 1



#####
# Get Records
#####

# Base path where the pre-generated structured records (.npy) are stored
path         = "../Data/"

# Base filename (without extension) of the records file to load
records_name = "records_CN_2_5_FN_0_1_FT_1_5_NS_10000_NE_10_ni_8_20250902-1520"

# Load records (structured NumPy array); allow_pickle=True because the dtype contains object fields (e.g., A, x, b arrays)
records      = np.load(file         = path + records_name + ".npy",
                       allow_pickle = True                        )


#####
# select relevant indices with "cond_numbr" == X and flag_neg == 0 or 1
#####

# Filter indices matching the requested condition number and sign-flag
indices      = np.where((records["cond_numbr"] == cond_numbr_global) & (records["flag_neg"] == flag_neg_global))[0]

# Subselect only the matching records
records      = records[indices]


#####
# Print
#####

# Console metadata to document the current run configuration
print("")
print("Records name:       ", records_name     )
print("")
print("Conditional Number: ", cond_numbr_global)
print("")
print("Flag negative:      ", flag_neg_global  )
print("")


#####
# start
#####

# Timestamp the start of this post-processing / execution run
start        = datetime.now()

# Iterate over the filtered records (each record corresponds to one (A,b,x,threshold,...) configuration)
for i in range(len(records)):

    # Print per-record sweep metadata
    print("")
    print("Conditional Number: ", records["cond_numbr"      ][i])
    print("")
    print("Factor Threshold:   ", records["factor_threshold"][i])
    print("")


    # Accumulators for K=10 repeated runs (PL ideal) to compute mean/std afterwards
    x_norm2_PL_ideal_array        = []
    b_norm2_PL_ideal_array        = []
    N_eff_PL_ideal_array          = []

    # Accumulators for K=10 repeated runs (QS ideal) to compute mean/std afterwards
    x_norm2_QS_ideal_array        = []
    b_norm2_QS_ideal_array        = []
    N_eff_QS_ideal_array          = []

    # Accumulators for K=10 repeated runs (QS Aachen transpiled/ideal) to compute mean/std afterwards
    x_norm2_QS_Aachen_ideal_array = []
    b_norm2_QS_Aachen_ideal_array = []
    N_eff_QS_Aachen_ideal_array   = []

    # Accumulators for K=10 repeated runs (QS Aachen noisy simulation) to compute mean/std afterwards
    x_norm2_QS_Aachen_sim_array   = []
    b_norm2_QS_Aachen_sim_array   = []
    N_eff_QS_Aachen_sim_array     = []


    # Repeat each configuration K=10 times to populate per-run fields and summary statistics
    for x in range(1,11):

        # Create/initialize IBM Quantum Runtime service (credentials + instance)
        # Note: avoid hardcoding tokens in committed code; use environment variables or secured config for submission.
        service = QiskitRuntimeService(channel  = 'ibm_cloud', 
                                       instance = "<INSTANCE>",
                                       token    = '<REDACTED_TOKEN>')


        #####
        # PL ideal
        #####

        # Run HHL using the "PL" package in an ideal simulator setting for a single (A,b) instance
        wrapper_1xA_1xb_PL_ideal        = wrapper_HHL(package        = "PL"                      ,
                                                      simulator      = "ideal"                   ,
                                                      modus          = "1xA_1xb"                 ,
                                                      A1             = records["A"          ][i] ,
                                                      b1             = records["b"          ][i] ,
                                                      n_shots        = int(records["n_shots"][i]),
                                                      print_circuit  = 0                         ,
                                                      print_solution = 1                         ,
                                                      factor_t       = 0.99                      ,
                                                      flag_use_gpu   = flag_use_gpu              )

        # Execute circuit construction + sampling + postprocessing (implementation encapsulated in wrapper_HHL)
        wrapper_1xA_1xb_PL_ideal.run()

        # Extract solution metrics:
        # - x_norm2: ||x||^2 (or proxy thereof, as defined in wrapper)
        # - b_norm2: ||b||^2 (or proxy thereof)
        # - N_eff: effective number of shots (implementation-defined)
        x_norm2_PL_ideal, b_norm2_PL_ideal, N_eff_PL_ideal = wrapper_1xA_1xb_PL_ideal.get_solution()

        # Store for summary stats
        x_norm2_PL_ideal_array.append(x_norm2_PL_ideal)
        b_norm2_PL_ideal_array.append(b_norm2_PL_ideal)
        N_eff_PL_ideal_array.append(  N_eff_PL_ideal  )

        # Store per-repeat values directly into the structured record (fields preallocated upstream)
        records["x_norm2_PL_ideal_" + str(x)][i] = x_norm2_PL_ideal
        records["b_norm2_PL_ideal_" + str(x)][i] = b_norm2_PL_ideal
        records["N_eff_PL_ideal_"   + str(x)][i] = N_eff_PL_ideal


        #####
        # QS ideal
        #####

        # Run HHL using the "QS" package in ideal simulator mode (no real backend, no noise)
        wrapper_1xA_1xb_QS_ideal        = wrapper_HHL(package        = "QS"                      ,
                                                      simulator      = "ideal"                   ,
                                                      flag_real_qc   = 0                         ,
                                                      modus          = "1xA_1xb"                 ,
                                                      A1             = records["A"          ][i] ,
                                                      b1             = records["b"          ][i] ,
                                                      n_shots        = int(records["n_shots"][i]),
                                                      print_circuit  = 0                         ,
                                                      print_solution = 1                         ,
                                                      factor_t       = 0.99                      ,
                                                      flag_use_gpu   = flag_use_gpu              )

        # Execute run and extract metrics
        wrapper_1xA_1xb_QS_ideal.run()

        x_norm2_QS_ideal, b_norm2_QS_ideal, N_eff_QS_ideal = wrapper_1xA_1xb_QS_ideal.get_solution()

        # Store for summary stats
        x_norm2_QS_ideal_array.append(x_norm2_QS_ideal)
        b_norm2_QS_ideal_array.append(b_norm2_QS_ideal)
        N_eff_QS_ideal_array.append(  N_eff_QS_ideal  )

        # Store per-repeat values into structured record
        records["x_norm2_QS_ideal_" + str(x)][i] = x_norm2_QS_ideal
        records["b_norm2_QS_ideal_" + str(x)][i] = b_norm2_QS_ideal
        records["N_eff_QS_ideal_"   + str(x)][i] = N_eff_QS_ideal


        #####
        # QS Aachen ideal
        #####

        # Run HHL using QS package with transpilation targeting the "ibm_aachen" backend,
        # but still running in ideal mode (flag_real_qc=0, simulator="ideal")
        wrapper_1xA_1xb_QS_Aachen_ideal = wrapper_HHL(package        = "QS"                      ,
                                                      simulator      = "ideal"                   ,
                                                      service        = service                   ,
                                                      backend        = "ibm_aachen"              ,
                                                      transpile      = "qc"                      ,
                                                      flag_real_qc   = 0                         ,
                                                      modus          = "1xA_1xb"                 ,
                                                      A1             = records["A"          ][i] ,
                                                      b1             = records["b"          ][i] ,
                                                      n_shots        = int(records["n_shots"][i]),
                                                      print_circuit  = 0                         ,
                                                      print_solution = 1                         ,
                                                      factor_t       = 0.99                      ,
                                                      flag_use_gpu   = flag_use_gpu              )
        
        # Execute run and extract metrics
        wrapper_1xA_1xb_QS_Aachen_ideal.run()

        x_norm2_QS_Aachen_ideal, b_norm2_QS_Aachen_ideal, N_eff_QS_Aachen_ideal = wrapper_1xA_1xb_QS_Aachen_ideal.get_solution()

        # Store for summary stats
        x_norm2_QS_Aachen_ideal_array.append(x_norm2_QS_Aachen_ideal)
        b_norm2_QS_Aachen_ideal_array.append(b_norm2_QS_Aachen_ideal)
        N_eff_QS_Aachen_ideal_array.append(  N_eff_QS_Aachen_ideal  )

        # Store per-repeat values into structured record
        records["x_norm2_QS_Aachen_ideal_" + str(x)][i] = x_norm2_QS_Aachen_ideal
        records["b_norm2_QS_Aachen_ideal_" + str(x)][i] = b_norm2_QS_Aachen_ideal
        records["N_eff_QS_Aachen_ideal_"   + str(x)][i] = N_eff_QS_Aachen_ideal


        #####
        # QS Aachen noisy
        #####

        # Run HHL using QS package with backend-specific transpilation,
        # and include a noise model (simulator="noisy") while still not executing on real hardware (flag_real_qc=0)
        wrapper_1xA_1xb_QS_Aachen_noisy = wrapper_HHL(package        = "QS"                      ,
                                                      simulator      = "noisy"                   ,
                                                      service        = service                   ,
                                                      backend        = "ibm_aachen"              ,
                                                      transpile      = "qc"                      ,
                                                      flag_real_qc   = 0                         ,
                                                      modus          = "1xA_1xb"                 ,
                                                      A1             = records["A"          ][i] ,
                                                      b1             = records["b"          ][i] ,
                                                      n_shots        = int(records["n_shots"][i]),
                                                      print_circuit  = 0                         ,
                                                      print_solution = 1                         ,
                                                      factor_t       = 0.99                      ,
                                                      flag_use_gpu   = flag_use_gpu              )

        # Execute run and extract metrics
        wrapper_1xA_1xb_QS_Aachen_noisy.run()

        x_norm2_QS_Aachen_noisy, b_norm2_QS_Aachen_noisy, N_eff_QS_Aachen_noisy = wrapper_1xA_1xb_QS_Aachen_noisy.get_solution()

        # Store for summary stats (note: naming uses *_sim_array although variable is "noisy" here)
        x_norm2_QS_Aachen_sim_array.append(x_norm2_QS_Aachen_noisy)
        b_norm2_QS_Aachen_sim_array.append(b_norm2_QS_Aachen_noisy)
        N_eff_QS_Aachen_sim_array.append(  N_eff_QS_Aachen_noisy  )

        # Store per-repeat values into structured record
        records["x_norm2_QS_Aachen_sim_" + str(x)][i] = x_norm2_QS_Aachen_noisy
        records["b_norm2_QS_Aachen_sim_" + str(x)][i] = b_norm2_QS_Aachen_noisy
        records["N_eff_QS_Aachen_sim_"   + str(x)][i] = N_eff_QS_Aachen_noisy


    # Aggregate summary statistics over the K repeats (mean/std) and store in the structured record
    records["N_eff_mean_PL_ideal"         ][i] = np.mean(  N_eff_PL_ideal_array       )
    records["N_eff_std_PL_ideal"          ][i] = np.std(   N_eff_PL_ideal_array       )
    records["x_norm2_mean_PL_ideal"       ][i] = np.mean(x_norm2_PL_ideal_array       )
    records["x_norm2_std_PL_ideal"        ][i] = np.std( x_norm2_PL_ideal_array       )
    records["b_norm2_mean_PL_ideal"       ][i] = np.mean(b_norm2_PL_ideal_array       )
    records["b_norm2_std_PL_ideal"        ][i] = np.std( b_norm2_PL_ideal_array       )

    records["N_eff_mean_QS_ideal"         ][i] = np.mean(  N_eff_QS_ideal_array       )
    records["N_eff_std_QS_ideal"          ][i] = np.std(   N_eff_QS_ideal_array       )
    records["x_norm2_mean_QS_ideal"       ][i] = np.mean(x_norm2_QS_ideal_array       )
    records["x_norm2_std_QS_ideal"        ][i] = np.std( x_norm2_QS_ideal_array       )
    records["b_norm2_mean_QS_ideal"       ][i] = np.mean(b_norm2_QS_ideal_array       )
    records["b_norm2_std_QS_ideal"        ][i] = np.std( b_norm2_QS_ideal_array       )

    records["N_eff_mean_QS_Aachen_ideal"  ][i] = np.mean(  N_eff_QS_Aachen_ideal_array)
    records["N_eff_std_QS_Aachen_ideal"   ][i] = np.std(   N_eff_QS_Aachen_ideal_array)
    records["x_norm2_mean_QS_Aachen_ideal"][i] = np.mean(x_norm2_QS_Aachen_ideal_array)
    records["x_norm2_std_QS_Aachen_ideal" ][i] = np.std( x_norm2_QS_Aachen_ideal_array)
    records["b_norm2_mean_QS_Aachen_ideal"][i] = np.mean(b_norm2_QS_Aachen_ideal_array)
    records["b_norm2_std_QS_Aachen_ideal" ][i] = np.std( b_norm2_QS_Aachen_ideal_array)

    records["N_eff_mean_QS_Aachen_sim"    ][i] = np.mean(  N_eff_QS_Aachen_sim_array  )
    records["N_eff_std_QS_Aachen_sim"     ][i] = np.std(   N_eff_QS_Aachen_sim_array  )
    records["x_norm2_mean_QS_Aachen_sim"  ][i] = np.mean(x_norm2_QS_Aachen_sim_array  )
    records["x_norm2_std_QS_Aachen_sim"   ][i] = np.std( x_norm2_QS_Aachen_sim_array  )
    records["b_norm2_mean_QS_Aachen_sim"  ][i] = np.mean(b_norm2_QS_Aachen_sim_array  )
    records["b_norm2_std_QS_Aachen_sim"   ][i] = np.std( b_norm2_QS_Aachen_sim_array  )



print(f" ")
# Print the last processed record (i is the last loop index if len(records)>0)
print("records[i]: ", records[i])

# Timestamp end of run
end = datetime.now()

####
# Save records
####
 
# Save the updated records array with a filename that encodes the selected CN/FN and a timestamp for traceability
np.save(path + records_name + "_CN_" + str(cond_numbr_global) + "_FN_" + str(flag_neg_global) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".npy", records)


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

# Echo configuration again at end of run for easy log scanning
print("")
print("Records Name:       ", records_name     )
print("")
print("Conditional Number: ", cond_numbr_global)
print("")
print("Flag negative:      ", flag_neg_global  )
print("")
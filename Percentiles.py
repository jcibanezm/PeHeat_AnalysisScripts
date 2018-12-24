import pickle
import numpy as np
from mpi4py import MPI
import time

start_time = time.time()

# Initialize MPI.
comm      = MPI.COMM_WORLD
pos       = comm.Get_rank()
num_procs = comm.Get_size()

#########################################################################
# User defined variables
#########################################################################

filename = "TotaHeating_parametric_50pcent.pkl"
data_dir = "/home/jcibanezm/codes/run/PeHeat/ParametricHeating"

#outdir = "/Users/juan/codes/run/PeHeating"                         # Laptop
outdir  = "/home/jcibanezm/codes/run/PeHeat/ParametricHeating" # DustBox
#outdir = "/data/gamera/jcibanezm/DustAnalysis"                    # Daikaiju
#outdir = "/data/gamera/jcibanezm/DustAnalysis"                    # Odin

outname = "Percentiles_50pcent.pkl"

percent = 100.

#########################################################################

if pos == 0:
    print("")
    print("===========================================================================")
    print(" Running the calculation of the percentiles of the photoelectric heating")
    print("===========================================================================")
    print("")

pkl_file = open("%s/%s"%(data_dir, filename), "rb")
PeH_dict = pickle.load(pkl_file)
pkl_file.close

def get_percentiles(Gtot, temp, ne, amin=3.5, amax=2500, numint=100):
    """
    This function finds the grain size where the 50 and 95 percentile of the photoelectric heating has been reached.
    
    Integrates the total photoelectric heating rate using the trapezium method and checks if the percentile has been reached.

    Returns:
        size50perc, size95perc.
    """
    import dust_size_dist as sizedist
    import numpy as np
    import parametric_PeHeat as parPeH
    
    #Gammatot = parPeH.get_Gamma_tot(Gtot, temp, ne, amin=amin, amax=amax)
    
    x = np.logspace(np.log10(amin), np.log10(amax), num=numint)
    yc = sizedist.dnda(x, "carbonaceous")
    ys = sizedist.dnda(x, "silicate")
    
    y_Gpe_s = np.zeros(numint, dtype=np.float)
    y_Gpe_c = np.zeros(numint, dtype=np.float)

    for j in range(numint):
        #print(j)
        y_Gpe_s[j] = parPeH.get_Gamma_dot(Gtot, temp, ne, x[j], "silicate")*sizedist.dnda(x[j], "silicate")*1.0e21
        y_Gpe_c[j] = parPeH.get_Gamma_dot(Gtot, temp, ne, x[j], "carbonaceous")*sizedist.dnda(x[j], "carbonaceous")*1.0e21    
        
    Gamma_tot_k = 0
    k = 1
    
    totd_s_fix = np.trapz(y_Gpe_s, x)
    totd_c_fix = np.trapz(y_Gpe_c, x)
    
    Gammatot_trapz = totd_s_fix + totd_c_fix
    
    #print("Using Trapezium rule:", (totd_s_fix+totd_c_fix)/1.0e21)
    
    perc25 = False
    perc50 = False    
    perc75 = False
    perc95 = False

    asize25perc = -1
    asize50perc = -1
    asize75perc = -1
    asize95perc = -1
    
    while perc95 == False:
        deltax = x[k] - x[k-1]
        Gamma_tot_k += (y_Gpe_s[k-1] + y_Gpe_s[k])/2.0*deltax + (y_Gpe_c[k-1] + y_Gpe_c[k])/2.0*deltax

        # look for the 50th percentile
        if (Gamma_tot_k) >= 0.25*Gammatot_trapz and perc25 == False:
            asize25perc = x[k]
            perc25 = True
        
        # look for the 50th percentile
        if (Gamma_tot_k) >= 0.5*Gammatot_trapz and perc50 == False:
            asize50perc = x[k]
            perc50 = True
            #print("Found the 50th percentile. size", x[k])
            #print("Gammatot = %.2g"%Gammatot_trapz)
            #print("Current PeHeat = %.2g"%Gamma_tot_k)
            
        # look for the 50th percentile
        if (Gamma_tot_k) >= 0.75*Gammatot_trapz and perc75 == False:
            asize75perc = x[k]
            perc75 = True

        # look for the 95th percentile
        if (Gamma_tot_k) >= 0.95*Gammatot_trapz and perc95 == False:
            asize95perc = x[k]
            perc95 = True
            #print("Found the 95th percentile. size", x[k])
            #print("Gammatot = %.2g"%Gammatot_trapz)
            #print("Current PeHeat = %.2g"%Gamma_tot_k)

        if k==numint-1:
            print("Did not find the 95th percentile!!!")
            print("Gammatot = %.2g"%Gammatot)
            print("Current PeHeat = %.2g"%Gamma_tot_k)
            break
            
        k+=1
                
    return asize25perc, asize50perc, asize75perc, asize95perc

ncells          = len(PeH_dict["nH"])
nsub            = np.int(ncells * percent/100.)
cells_per_proc =  nsub // num_procs

if pos == 0:
    print("Number of processors = %i"%num_procs)
    print("Cells per processor  = %i"%cells_per_proc)
    print("Total number of cells= %i"%nsub)

if pos == num_procs-1:
    if (pos+1)*cells_per_proc < nsub:
        missing_cells = (nsub - (pos+1)*cells_per_proc)
        cells_per_proc += missing_cells
        print("Adding %i cells to the last processor"%missing_cells)

Gtot = np.array(PeH_dict["Gtot"][pos*cells_per_proc:(pos+1)*cells_per_proc])
temp = np.array(PeH_dict["temp"][pos*cells_per_proc:(pos+1)*cells_per_proc])
ne   = np.array(PeH_dict["new_ne"][pos*cells_per_proc:(pos+1)*cells_per_proc])

size25 = np.zeros(cells_per_proc, dtype=np.float)
size50 = np.zeros(cells_per_proc, dtype=np.float)
size75 = np.zeros(cells_per_proc, dtype=np.float)
size95 = np.zeros(cells_per_proc, dtype=np.float)

for i in range(cells_per_proc):
    # Calculate the percentiles.
    size25[i], size50[i], size75[i], size95[i] = get_percentiles(PeH_dict["Gtot"][i], PeH_dict["temp"][i], PeH_dict["new_ne"][i])
    if pos == 0:
        if (i*10.%cells_per_proc==0):
            print("Progress %i"%(i*100.//cells_per_proc))

# Gather all the results to proc 0.
results25 = comm.gather(size25, root=0)
results50 = comm.gather(size50, root=0)
results75 = comm.gather(size75, root=0)
results95 = comm.gather(size95, root=0)


# Append the dictionaries and save the file.
if pos == 0:
    
    # Loop over all procesors to concatenate the data.
    for proc in range(num_procs-1):
        size25 = np.append(size25, results25[proc+1])
        size50 = np.append(size50, results50[proc+1])
        size75 = np.append(size75, results75[proc+1])
        size95 = np.append(size95, results95[proc+1])

    percDict = {"info":"Dictionary with stuff"}
    percDict["size25"] = size25
    percDict["size50"] = size50
    percDict["size75"] = size75
    percDict["size95"] = size95

    percDict["Gtot"] = PeH_dict["Gtot"]
    percDict["nH"]   = PeH_dict["nH"]
    percDict["temp"] = PeH_dict["temp"]
    percDict["new_ne"]= PeH_dict["new_ne"]

    # Now save the dict
    #outname = "fz_%.4iAA_%s_CR_%s.pkl"%(grain_size, grain_type, include_CR)
    outfile = open('%s/%s'%(outdir, outname), 'wb')
    pickle.dump(percDict, outfile)
    outfile.close()
    end_time = time.time()
    print("")
    print("------------------ Done ----------------------")
    print("Saving calculation of percentiles %s/%s"%(outdir, outname))
    print("Number of cells sampled: %i"%nsub)
    print("Number of cores used:    %i"%num_procs)
    print("Time per core (hours):   %.2f"%((end_time - start_time)/3600.))
    print("Total CPU hours:         %.2f"%((end_time - start_time)*num_procs/3600.))
    print("----------------------------------------------")
    print("")


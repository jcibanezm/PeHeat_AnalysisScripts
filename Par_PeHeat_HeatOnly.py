from mpi4py import MPI

import yt
import cPickle as pickle
import numpy as np
import time

import parametric_PeHeat as parPeH

start_time = time.time()
comm = MPI.COMM_WORLD

pos       = comm.Get_rank()
num_procs = comm.Get_size()


###################################################################################
# Changeable variables
###################################################################################
percent    = 5.

# Where the dictionary is going to be saved
#outdir = "/Users/juan/codes/run/PeHeating"                         # Laptop
outdir  = "/home/jcibanezm/codes/run/PeHeat/ParametricHeating" # DustBox
#outdir = "/data/gamera/jcibanezm/DustAnalysis"                    # Daikaiju
#outdir = "/data/gamera/jcibanezm/DustAnalysis"                    # Odin

outname = "HeatingOnly_parametric_%ipcent.pkl"%(percent)

###################################################################################


filename = "TotaHeating_parametric_50pcent.pkl"
data_dir = "/home/jcibanezm/codes/run/PeHeat/ParametricHeating/" #DustBox
#data_dir = "/Users/juan/codes/run/PeHeating/" #Laptop

pkl_file = open("%s%s"%(data_dir, filename))
heatDict = pickle.load(pkl_file)
pkl_file.close

too_high = np.where(heatDict["nH"] > 1.0e6)
for key in heatDict.keys():
    #arrs.append(0)
    if key!="info":
        #print(key)
        heatDict[key] = np.delete(heatDict[key], too_high)


##################### Randomly choose the cells to analyze ######################
np.random.seed(1)

ncells         = len(heatDict["nH"])
n5             = np.int(ncells * percent/100.)
rand_index     = np.random.randint(0, ncells, n5)
cells_per_proc = n5 // num_procs

if pos == 0:
    print("Number of processors = %i"%num_procs)
    print("Cells per processor  = %i"%cells_per_proc)
    #print("Total number of cells= %i"%nsub)

if pos == num_procs:
    if (pos+1)*cells_per_proc < n5:
        missing_cells = (n5 - (pos+1)*cells_per_proc)
        cells_per_proc += missing_cells
        print("Adding %i cells to the last processor"%missing_cells)

temp = np.array(heatDict["temp"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
nH   = np.array(heatDict["nH"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
nH2  = np.array(heatDict["nH2"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
ne   = np.array(heatDict["ne"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
xe   = np.array(heatDict["xe"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
#xH2  = np.array(heatDict["xH2"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
#Av   = np.array(heatDict["cdto"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
cell_mass = np.array(heatDict["cell_mass"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
NH2  = np.array(heatDict["NH2"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
Ntot  = np.array(heatDict["Ntot"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
zeta = np.array(heatDict["zeta"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
new_ne = np.array(heatDict["new_ne"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
new_xe = np.array(heatDict["new_xe"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
G_CR = np.array(heatDict["G_CR"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
Gtot = np.array(heatDict["Gtot"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
Geff = np.array(heatDict["Geff"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
Av = np.array(heatDict["Av"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])


PeHtot = np.zeros_like(nH, dtype=np.float)
for i in range(len(nH)):
    PeHtot[i] = nH[i]*parPeH.get_Gamma_tot_heatonly(Gtot[i], temp[i], new_ne[i])

# Add the calculation of the 50 and 95 percentile here!!!

HeatDict = {"info": "Dictionary "}
HeatDict["nH"]     = nH
HeatDict["nH2"]    = nH2
HeatDict["NH2"]    = NH2
HeatDict["Ntot"]   = Ntot
HeatDict["temp"]   = temp
HeatDict["ne"]     = ne
HeatDict["new_ne"] = new_ne
HeatDict["xe"]     = xe
HeatDict["new_xe"] = new_xe
HeatDict["Av"]     = Av
HeatDict["Geff"]   = Geff
HeatDict["G_CR"]   = G_CR
HeatDict["Gtot"]   = Gtot
HeatDict["PeHtot"] = PeHtot
HeatDict["cell_mass"] = cell_mass
HeatDict["zeta"]   = zeta

# Gather all the results to proc 0.
results = comm.gather(HeatDict, root=0)

# Append the dictionaries and save the file.
if pos == 0:
    peHDict        = dict(results[0])
    peHDict["MPI"] = "MPI calculation of the photoelectric heating in %i procs" %(num_procs)
    
    # Loop over all procesors to concatenate the data.
    for proc in range(num_procs-1):
        # Loop over al fields.
        for field in HeatDict.keys():
            if field != "info":
                # In the new dictionary, append the data of a given field from another processor.
                peHDict[field] = np.append(peHDict[field], results[proc+1][field])

    # Now save the dict
    #outname = "fz_%.4iAA_%s_CR_%s.pkl"%(grain_size, grain_type, include_CR)
    outfile = open('%s/%s'%(outdir, outname), 'wb')
    pickle.dump(peHDict, outfile)
    outfile.close()
    end_time = time.time()
    print("")
    print("------------------ Done ----------------------")
    print("Saving total calculation of the photoelectric heating %s/%s"%(outdir, outname))
    print("Number of cells sampled: %i"%n5)
    print("Number of cores used:    %i"%num_procs)
    print("Time per core (hours):   %.2f"%((end_time - start_time)/3600.))
    print("Total CPU hours:         %.2f"%((end_time - start_time)*num_procs/3600.))
    print("----------------------------------------------")
    print("")

from mpi4py import MPI
from yt.units import pc, kpc, second, Kelvin, gram, erg, cm

import yt
import cPickle as pickle
import numpy as np
import fzMPI
import time

import PeHeat_Functions as peh
import parametric_fz as fzpar
import parametric_PeHeat as parPeH
from pynverse import inversefunc
import compute_charge_dist as fz

start_time = time.time()
comm = MPI.COMM_WORLD

pos       = comm.Get_rank()
num_procs = comm.Get_size()


###################################################################################
# Changeable variables
###################################################################################
percent    = 0.1

# Where the flash data is located
data_dir   = "/Users/juan/codes/run/Silcc/CF_Prabesh"      # Laptop
#data_dir   = "/home/jcibanezm/codes/run/Silcc/CF_Prabesh" # DustBox
#data_dir = "/data/gamera/jcibanezm/DustAnalysis/CF_Data"  # Daikaiju
#data_dir = "/data/gamera/jcibanezm/DustAnalysis/CF_Data"  # Odin

filename   = data_dir + "/NL99_R8_cf_hdf5_chk_0028"

# Where the dictionary is going to be saved
outdir = "/Users/juan/codes/run/PeHeating"                         # Laptop
#outdir  = "/home/jcibanezm/codes/run/ChargeStatisticsAnalysis/CR" # DustBox
#outdir = "/data/gamera/jcibanezm/DustAnalysis"                    # Daikaiju
#outdir = "/data/gamera/jcibanezm/DustAnalysis"                    # Odin

outname = "TotaHeating.pkl"

###################################################################################
#
#
#
# Define some constant parameters to be used.
mp      = 1.6726e-24  * gram # g
mH      = 1.6733e-24  * gram
mC      = 12.011*mH
#mu      = 1.2924
kb      = 1.3806e-16  *erg / Kelvin # erg K-1
GNewton = 6.6743e-8   * cm**3 / (gram * second**2 )# cm3 g-1 s-2
Msun    = 1.9884e33   * gram
#mm      = mu*mp

G0         = 1.7

ppc = 3.0856776e18

# -------------------------------------------------------------
#              Create a lot of new derived fields
# -------------------------------------------------------------

# Create a derived field.
# Hydrogen number density
def numdensH(field, data):
    nH = data["dens"]*(data["ihp "]+data["iha "]+data["ih2 "])/(1.4*mH)
    return nH

# Molecular Hydrogen number density
def numdensH2(field, data):
    nH2 = data["dens"]*(data["ih2 "])/(1.4*mH)
    return nH2

# Carbon number density
def numdensC(field, data):
    nC = data["dens"]*(data["icp "]+data["ico "])/(1.4*mC)
    return nC

# electron number density
def numdense(field, data):
    ne = data["dens"]*(data["ihp "]/(1.4*mH) + data["icp "]/(1.4*mC))
    return ne

# Ionized hydrogen fraction
def xHp(field, data):
    nH  = data["dens"]*(data["ihp "]+data["iha "]+data["ih2 "])/(1.4*mH)
    xHp = data["dens"]*data["ihp "]/(1.4*mH)
    xHp = xHp / nH
    return xHp

# Molecular hydrogen fraction
def xH2(field, data):
    nH  = data["dens"]*(data["ihp "]+data["iha "]+data["ih2 "])/(1.4*mH)
    xH2 = data["dens"]*data["ih2 "]/(1.4*mH)
    xH2 = xH2 / nH
    return xH2

# Ionized carbon fraction
def xCp(field, data):
    nC  = data["dens"]*(data["icp "]+data["ico "])/(1.4*mC)
    xCp = data["dens"]*data["icp "]/(1.4*mC) / nC
    return xCp

# electron fraction
def xe(field, data):
    nH = data["dens"]*(data["ihp "]+data["iha "]+data["ih2 "])/(1.4*mH)
    nC = data["dens"]*(data["icp "]+data["ico "])/(1.4*mC)
    ne = data["dens"]*(data["ihp "]/(1.4*mH) + data["icp "]/(1.4*mC))
    xe = ne / (nH + nC)
    return xe

yt.add_field('nH', function=numdensH,  units="1/cm**3", force_override=True)
yt.add_field('nH2',function=numdensH2, units="1/cm**3", force_override=True)
yt.add_field('nC', function=numdensC,  units="1/cm**3", force_override=True)
yt.add_field('ne', function=numdense,  units="1/cm**3", force_override=True)
yt.add_field('xHp', function=xHp,      units="dimensionless", force_override=True)
yt.add_field('xH2', function=xH2,      units="dimensionless", force_override=True)
yt.add_field('xCp', function=xCp,      units="dimensionless", force_override=True)
yt.add_field('xe', function=xe,        units="dimensionless", force_override=True)
#yt.add_field('G',  function=GG,        units="dimensionless", force_override=True)

pf = yt.load("%s"%(filename))


c  = [0,0,0]
le = [-4.93696000e+19, -4.93696000e+19, -4.93696000e+19]
re = [ 4.93696000e+19,  4.93696000e+19,  4.93696000e+19]

box = pf.box(le, re)


min_dens = np.min(box["density"])
max_dens = np.max(box["density"])

min_nh   = np.min(box["nH"])
max_nh   = np.max(box["nH"])

min_temp = np.min(box["temperature"])
max_temp = np.max(box["temperature"])

min_ne = np.min(box["ne"])
max_ne = np.max(box["ne"])

min_xe = np.min(box["xe"])
max_xe = np.max(box["xe"])

min_Av = np.min(box["cdto"])
max_Av = np.max(box["cdto"])


if pos == 0:
    print("-----------------------------------------------------------")
    print("Some properties of the simulation:")
    print("Density,     min = %.2g,\t max = %.2g"%(min_dens, max_dens))
    print("Temperature, min = %.2g,\t\t max = %.2g"%(min_temp, max_temp))
    print("ndens H,     min = %.2g,\t max = %.2g"%(min_nh, max_nh))
    print("ndens e,     min = %.2g,\t max = %.2g"%(min_ne, max_ne))
    print("e fraction,  min = %.2g,\t max = %.2g"%(min_xe, max_xe))
    print("Av           min = %.2g,\t max = %.2g"%(min_Av, max_Av))

##################### Randomly choose the cells to analyze ######################
np.random.seed(1)

ncells         = len(box["nH"])
n5             = np.int(ncells * percent/100.)
rand_index     = np.random.randint(0, ncells, n5)
cells_per_proc = n5 // num_procs

if pos == num_procs:
    if (pos+1)*cells_per_proc < n5:
        missing_cells = (n5 - (pos+1)*cells_per_proc)
        cells_per_proc += missing_cells
        print("Adding %i cells to the last processor"%missing_cells)

temp = np.array(box["temp"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
nH   = np.array(box["nH"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
nH2  = np.array(box["nH2"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
ne   = np.array(box["ne"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
xe   = np.array(box["xe"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
xH2  = np.array(box["xH2"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
Av   = np.array(box["cdto"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
fH2shield = np.array(box["cdh2"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])

Ntot = Av * 1.87e21
Geff = G0*np.exp(-2.5*Av)

new_ne = np.zeros_like(nH)
new_xe = np.zeros_like(nH)

cell_mass = np.array(box["cell_mass"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]].in_units("Msun"))

NH2    = np.zeros_like(nH, dtype=np.float)
zeta   = np.zeros_like(nH, dtype=np.float)
new_ne = np.zeros_like(nH, dtype=np.float)
new_xe = np.zeros_like(nH, dtype=np.float)
G_CR   = np.zeros_like(nH, dtype=np.float)

for i in range(len(nH)):
    NH2[i]   = inversefunc(fz.get_f2shield, y_values=fH2shield[i], args=(temp[i]))
    zeta[i]  = fz.get_zeta(NH2[i])
    new_ne[i], new_xe[i] = fz.compute_new_xe([nH[i], 0.], [xe[i],0], xH2[i], zeta[i])
    G_CR[i]  = fz.get_G_CR(NH2[i])

Gtot = G_CR + Geff

PeHtot = np.zeros_like(nH, dtype=np.float)
for i in range(len(nH)):
    PeHtot[i] = nH[i]*parPeH.get_Gamma_tot(Gtot[i], temp[i], new_ne[i])

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
HeatDict["fH2shield"] = fH2shield
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

import compute_charge_dist as fz
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import PeHeat_Functions as peh
import parametric_fz as fzpar
import dust_size_dist as sizedist

import yt
from yt.units import pc, kpc, second, Kelvin, gram, erg, cm

##########################################################
# Input variables.
#data_dir   = "/home/jcibanezm/codes/run/Silcc/CF_Prabesh"
data_dir   = "/Users/juan/codes/run/Silcc/CF_Prabesh"

##########################################################

microntocm    = 1.0e-4
cmtomicron    = 1.0e4
AAtocm        = 1.0e-8
cmtoAA        = 1.0e8
microntoAA    = 1.0e4
AAtomicron    = 1.0e-4
ergtoeV       = 6.242e11
eVtoerg       = 1.602e-12

hplanck       = 4.135667662e-15 # eV s
clight        = 2.99792458e10   # cm s-1

# Define some constant parameters to be used.
mp      = 1.6726e-24  * gram # g
mH      = 1.6733e-24  * gram
mC      = 12.011*mH
#mu      = 1.2924
kb      = 1.3806e-16  *erg / Kelvin # erg K-1
GNewton = 6.6743e-8   * cm**3 / (gram * second**2 )# cm3 g-1 s-2
Msun    = 1.9884e33   * gram
#mm      = mu*mp

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

# electron fraction
#def GG(field, data): 
#    G = fz.get_G(data["cdto"], 1.68)
#    return G


yt.add_field('nH', function=numdensH,  units="1/cm**3", force_override=True)
yt.add_field('nH2',function=numdensH2, units="1/cm**3", force_override=True)
yt.add_field('nC', function=numdensC,  units="1/cm**3", force_override=True)
yt.add_field('ne', function=numdense,  units="1/cm**3", force_override=True)
yt.add_field('xHp', function=xHp,      units="dimensionless", force_override=True)
yt.add_field('xH2', function=xH2,      units="dimensionless", force_override=True)
yt.add_field('xCp', function=xCp,      units="dimensionless", force_override=True)
yt.add_field('xe', function=xe,        units="dimensionless", force_override=True)
#yt.add_field('G',  function=GG,        units="dimensionless", force_override=True)


filename   = data_dir + "/NL99_R8_cf_hdf5_chk_0028"

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


wnm = ((box["temperature"].in_units("K") > 6.0e3)&(box["temperature"].in_units("K") < 1.0e4)&(box["iha "] > 0.6)&(box["nH"] >= 0.75))
cnm = ((box["temperature"].in_units("K") > 50)   &(box["temperature"].in_units("K") < 70)   &(box["iha "] > 0.6))
cmm = ((box["temperature"].in_units("K") < 50)   &(box["ih2 "] > 0.6) &(box["nH"] <3.0e4))

wnmtrue = np.argwhere(wnm==True)
cnmtrue = np.argwhere(cnm==True)
cmmtrue = np.argwhere(cmm==True)

wnmindex = wnmtrue[0]
cnmindex = cnmtrue[0]
cmmindex = [3479445]


indexarr = np.array([wnmindex, cnmindex, cmmindex])

temp= np.array( box["temp"][indexarr])
dd  = np.array( box["dens"][indexarr])
nH  = np.array( box["nH"]  [indexarr])
nH2 = np.array( box["nH2"]  [indexarr])
nC  = np.array( box["nC"]  [indexarr])
ne  = np.array( box["ne"]  [indexarr])
xe  = np.array( box["xe"]  [indexarr])
xHp = np.array( box["xHp"]  [indexarr])
xH2 = np.array( box["xH2"]  [indexarr])
xCp = np.array( box["xCp"]  [indexarr])
Av  = np.array( box["cdto"][indexarr])
fH2shield  = np.array( box["cdh2"][indexarr])


###################################################

def netHeating_full(grain_size, grain_type, nH, temp, xe, xH2, Ntot, NH2, G0=1.7, save_output=False, outdir="default", pedantic=False):
    """
    Perform the full calculation of the net heating by a single grain given the ISM ambient parameters.
    """
    import cPickle as pickle
    import compute_charge_dist as fz
    import numpy as np
    import PeHeat_Functions as peh
    
    #Full calculation of the net heating by a grain at a given cell.
    Qabs = fz.get_QabsTable(grain_type, grain_size)

    #print("Running grain size %i, "%(grain_size))

    zeta = fz.get_zeta(NH2)

    # Compute the charge distribution.
    ############################################################################################
    Jpe, Je, Jh, Jc, ZZall = fz.compute_currents ([nH, nH*1.0e-4], [xe, 1.0e4*min(xe, 1.0e-4)], xH2, temp, zeta, grain_size, Ntot, grain_type, Qabs, G0=G0)
    JCRe, JCRpe, ZZnew     = fz.compute_CR_currents(nH, zeta, grain_size, grain_type, Qabs)

    zeq                    = fz.get_zeq_vec      (Jpe, Je, Jh, Jc, ZZall, grain_size, grain_type)
    new_zmin, new_zmax     = fz.get_new_zmin_zmax([nH, nH*1.0e-4], [xe, 1.0e4*min(xe, 1.0e-4)], temp, grain_size, Ntot, grain_type, Qabs, zeta, zeq=zeq, G0=G0, includeCR=True)    
    
    ffzCR, ZZfz            = fz.vector_fz        (Jpe, Je, Jh, Jc, JCRe, JCRpe, ZZall, new_zmin, new_zmax, includeCR=True)

    # Compute the minimum and maximum allowd charges by this grain
    Zmin, Zmax = fz.get_Zmin(grain_size, grain_type), fz.get_Zmax(grain_size, grain_type)
    Znum       = int(Zmax + abs(Zmin) +1)
    ZZ_all     = np.linspace(Zmin, Zmax, num=Znum)

    Gamma_dotdot_Z = np.zeros_like(ZZ_all, dtype=np.float)

    for i in range(Znum):
        Gamma_dotdot_Z[i] = peh.get_Gamma_pe_dotdot(grain_size, ZZ_all[i], grain_type, Ntot, Qabs)    

    Cooling = peh.Cool_per_Grain(grain_size, grain_type, ZZfz, ffzCR, nH, xe, temp)

    Av = Ntot/1.87e21
    
    Geff = G0*np.exp(-2.5*Av)
    G_CR = fz.get_G_CR(NH2)
    
    Gtot = Geff+G_CR
    
    Heating = peh.Gamma_per_grain(ZZ_all, Gamma_dotdot_Z, ZZfz, ffzCR)

    netHeating = Heating - Cooling
    
    if save_output:
        if nH < 1.0:
            phase = "WNM"
        elif nH>1.0 and nH<100:
            phase = "CNM"
        else:
            phase = "CMM"
        
        if outdir == "default":
            outdir = "/home/jcibanezm/codes/run/PeHeat"
        
        filename = "%s/TotalHeating_ISM%s_%s_%.3fAA.pkl"%(outdir, phase, grain_type, grain_size)
        
        dictionary = {"info":"Saving the Heating, Cooling, charge array and charge distribution."}
        dictionary["netHeating"] = netHeating
        dictionary["Heating"] = Heating
        dictionary["Cooling"] = Cooling
        dictionary["grain_size"] = grain_size
        dictionary["grain_type"] = grain_type
        dictionary["ffz"] = ffzCR
        dictionary["ZZ"] = ZZfz
        dictionary["nH"] = nH
        dictionary["temp"] = temp
        dictionary["Geff"] = Geff
        dictionary["Gtot"] = Gtot
        dictionary["zeta"] = zeta
        dictionary["Ntot"] = Ntot
        dictionary["NH2"] = NH2
        dictionary["ne"] = nH*xe
        dictionary["xe"] = xe
        dictionary["xH2"] = xH2
        dictionary["zmin"] = new_zmin
        dictionary["zmax"] = new_zmax

        if pedantic == True:
            print("Saving a file with the information of the net Heating and Cooling in this cell.")
            print("Cell properties:")
            print("grain size = %.1f, grain type = %s"%(grain_size, grain_type))
            print("ntot = %.2g \t temp = %.2g \t Geff = %.2g \t Gtot = %.2g"%(nH, temp, Geff, Gtot))
            print("zeta = %.2g \t Ntot = %.2g \t NH2 = %.2g \t ne = %.2g"%(zeta, Ntot, NH2, nH*xe))
            print("xe = %.2g \t xH2 = %.2g"%(xe, xH2))
            print("netHeating = %.2g erg s-1  \t Cooling %.2g erg s-1"%(netHeating, Cooling))
            print("ZZ =", ZZfz)
            print("f(Z) = ",ffzCR)
        
        outfile = open('%s'%(filename), 'wb')
        pickle.dump(dictionary, outfile)
        outfile.close()

    #return netHeating, Cooling, ZZfz, ffzCR
    return Heating


def get_Gamma_tot_Full(nH, temp, xe, xH2, Ntot, NH2, amin, amax, G0=1.7, numint=50, save_output=False, outdir="default"):
    """
    Returns GammaTot*1.0e21
    """
    
    import dust_size_dist as sizedist
    from scipy.integrate import simps
    
    x = np.logspace(np.log10(amin), np.log10(amax), num=numint)
    yc = sizedist.dnda(x, "carbonaceous")
    ys = sizedist.dnda(x, "silicate")

    for grain_size in x:
        y_Gpe_s = netHeating_full(grain_size, "silicate", nH, temp, xe, xH2, Ntot, NH2, G0=G0, save_output=save_output, outdir=outdir)*ys*1.0e21
        y_Gpe_c = netHeating_full(grain_size, "carbonaceous", nH, temp, xe, xH2, Ntot, NH2, G0=G0, save_output=save_output, outdir=outdir)*yc*1.0e21

    Gamma_pe_sil_fix  = simps(y_Gpe_s, x)
    Gamma_pe_carb_fix = simps(y_Gpe_c, x)
    
    Gamma_tot= Gamma_pe_sil_fix + Gamma_pe_carb_fix
    
    return Gamma_tot


Gtot_CNM = get_Gamma_tot_Full(nH[1], temp[1], xe[1], xH2[1], Ntot[1], NH2[1], 3.5, 2500, numint=50, save_output=True)
Gtot_CMM = get_Gamma_tot_Full(nH[2], temp[2], xe[2], xH2[2], Ntot[2], NH2[2], 3.5, 2500, numint=50, save_output=True)
Gtot_WNM = get_Gamma_tot_Full(nH[0], temp[0], xe[0], xH2[0], Ntot[0], NH2[0], 3.5, 2500, numint=50, save_output=True)


print(Gtot_CMM*1.0e-21)
print(Gtot_CNM*1.0e-21)
print(Gtot_WNM*1.0e-21)


#!/usr/bin/env python

import numpy as np
import scipy as sp
import pandas as pd
import datetime
from netCDF4 import Dataset
from scipy import interpolate
from pysolar.solar import *
from hapi import *
import math
import pickle
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
import spectral.io.envi as envi
from spectral import *
from scipy.special     import gamma
from scipy import stats
from scipy.optimize import minimize

from numba import jit

#Set random seed
np.random.seed(seed=200)

#RMSE function
rms_func = lambda x1, x2: np.sqrt(((x1 - x2) ** 2).mean())

#Class to dynamically add files to path
class make_class:
    pass

#Initialization function - pass a dictionary of arguments here
#that will be used to tell IMAP-DOAS where certain files and
#file metadata lives
def initialize(init_dict={}):

    #Set location of hyperspectral images
    files = make_class()
    coord_sub = make_class()
    set_inputs = make_class()
    set_flight = make_class()
    wave_class =  make_class()

    #Get dictionary names
    dict_names = list(init_dict.keys())
    
    #Define the folder where the results will be saved
    if 'rad_dir' in dict_names:
        files.rad_dir = init_dict['rad_dir']
    else:
        files.rad_dir = 'enp20180922t205612_rad'
        
    if 'rad_name' in dict_names:
        files.rad_name = init_dict['rad_name']
    else:
        files.rad_name = 'avr20180922t205612_rad'
    
    if 'main' in dict_names:
        files.main = init_dict['main']
    else:
        files.main = '/Users/cusworth/Documents/EnMAP/converted'
    
    #Files locations
    files.folder_data = files.main + '/' + files.rad_dir + '/'
    files.rad = files.folder_data + files.rad_name
    files.rad_head = files.rad + '.hdr'
    
    #Ancillary data
    if 'solar_spec' in dict_names:
        files.solar_spec = init_dict['solar_spec']
    else:
        files.solar_spec = 'ancillary/solar_741_2597nm_0pt01int.dat'
    
    if 'hitran_loaded' in dict_names:
        files.hitran_loaded = init_dict['hitran_loaded']
    else:
        files.hitran_loaded = False
        
    if 'wave_pos' in dict_names:
        files.wave_pos = init_dict['wave_pos']
    else:
        files.wave_pos = 'ancillary/av_wvl.txt'
    
    if 'met_file' in dict_names:
        files.met_file = init_dict['met_file']
    else:
        files.met_file = True


    #Select portion of scene to run IMAP on
    if 'ctmf_mask_include' in dict_names:
        files.ctmf_mask_include = init_dict['ctmf_mask_include']
    else:
        set_inputs.ctmf_mask_include = False 

    #If running on subscene
    if set_inputs.ctmf_mask_include:
        
        #Define coordinates of subset
        if 'coord_sub' in dict_names:
            
            coord_sub.include_x_left = init_dict['coord_sub'][0] 
            coord_sub.include_y_top = init_dict['coord_sub'][1]
            coord_sub.include_x_right = init_dict['coord_sub'][2]
            coord_sub.include_y_bottom = init_dict['coord_sub'][3]
            
        else:
            coord_sub.include_x_left = 0#300 
            coord_sub.include_y_top = 0#500
            coord_sub.include_x_right = 50#350    
            coord_sub.include_y_bottom = 50#550  
  
    #Set flight parameters
    if 'alt_agl_km' in dict_names:
        set_flight.alt_agl_km = init_dict['alt_agl_km']#653
    else:
        set_flight.alt_agl_km = 4
        
    if 'latitude' in dict_names:
        set_flight.latitude = init_dict['latitude']
    else:
        set_flight.latitude = 37.536295
        
    if 'longitude' in dict_names:
        set_flight.longitude = init_dict['longitude']
    else:
        set_flight.longitude = -120.952019
        
    if 'elev' in dict_names:
        set_flight.elev = init_dict['elev']
    else:
        set_flight.elev = 0.1
        
    if 'UTC' in dict_names:
        set_flight.UTC = init_dict['UTC']
    else:
        set_flight.UTC = -7
    
    #Get date information from filename
    set_flight.date_flight = int(files.rad_dir[3:11])
    set_flight.year = int(str(set_flight.date_flight)[0:4])
    set_flight.month = int(str(set_flight.date_flight)[4:6])
    set_flight.day = int(str(set_flight.date_flight)[6:8])
    set_flight.hour_utc = int(files.rad_dir[12:14])
    set_flight.min_utc = int(files.rad_dir[14:16])
    set_flight.sec_utc = int(files.rad_dir[16:18])

    #Other general parameters
    if 'fwhm' in dict_names:
        set_inputs.fwhm = init_dict['fwhm']
    else:
        set_inputs.fwhm = 5
    
    if 'SNR' in dict_names:
        set_inputs.SNR = init_dict['SNR']
    else:
        set_inputs.SNR = 180
    
    if 'deg_poly' in dict_names:
        set_inputs.deg_poly = init_dict['deg_poly']
    else:
        set_inputs.deg_poly = 4

    if 'do_legendre' in dict_names:
        set_inputs.do_legendre = init_dict['do_legendre']
    else:
        set_inputs.do_legendre = True

    #Wavelengths to query hitran and run retrieval
    if 'hitran_wvl' in dict_names:
        wave_class.hitran_wvl = init_dict['hitran_wvl']
    else:
        wave_class.hitran_wvl = [2200, 2450]
    
    if 'inversion_wvl' in dict_names:
        wave_class.inversion_wvl = init_dict['inversion_wvl']
    else:
        wave_class.inversion_wvl = [2215, 2415]
    
    return files, set_inputs, wave_class, set_flight, coord_sub


#Query MERRA-2 openDAP server to get meteorological data
#or load from a previous query
def load_merra(files, set_flight):

    if files.met_file == False:
        
        #Load libraries needed to access server
        import urllib.request as urllib2
        import http.cookiejar
        from pydap.client import open_url
        from pydap.cas.urs import setup_session
       
        
        #Must have correct authentication
        #NASA earthdata authenticationshould be stored in ~/.netrc file
        import netrc
        authData = netrc.netrc().hosts['urs.earthdata.nasa.gov']
        myLogin = authData[0]
        myPassword = authData[2]
        
        #Query MERRA data
        print('Querying MERRA data........')

        #date_str = str(set_flight.date_flight)[0:4] + str(set_flight.date_flight)[4:6] + str(set_flight.date_flight)[6:8]
        date_str = str(int(str(set_flight.date_flight)[0:4])-1) + str(set_flight.date_flight)[4:6] + str(set_flight.date_flight)[6:8] #dcusworth hack
        #Query OpenDAP
        base_url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov:443/opendap/MERRA2/M2I3NVASM.5.12.4/%YR%/%MON%/MERRA2_400.inst3_3d_asm_Nv.%DATE%.nc4'
        #iurl = base_url.replace('%YR%', str(set_flight.date_flight)[0:4])
        iurl = base_url.replace('%YR%', str(int(str(set_flight.date_flight)[0:4])-1)) #dcusworth hack
        iurl = iurl.replace('%MON%', str(set_flight.date_flight)[4:6])
        iurl = iurl.replace('%DATE%', date_str)
        session = setup_session(myLogin, myPassword, check_url = iurl)
        dataset = open_url(iurl, session=session)

        #Select metadata
        dlon = np.array(dataset['lon'][:])
        dlat = np.array(dataset['lat'][:])
        dtime = np.array(dataset['time'][:])

        #Select right dimensions
        sel_time = np.argmin(np.abs(set_flight.hour_utc - (dtime/60)))
        sel_lon = np.argmin(np.abs(dlon - set_flight.longitude))
        sel_lat = np.argmin(np.abs(dlat - set_flight.latitude))

        #Get needed variables
        print('Downloading surface pressure........')
        PS = np.array(dataset['PS'][:][sel_time, sel_lat, sel_lon])
        print('Downloading pressure levels........')
        PL = np.array(dataset['PL'][:][sel_time,:, sel_lat, sel_lon])
        print('Downloading specific humidity........')
        QV = np.array(dataset['QV'][:][sel_time,:, sel_lat, sel_lon])
        print('Downloading temperature profile........')
        T = np.array(dataset['T'][:][sel_time,:, sel_lat, sel_lon])
        print('Downloading pressure thickness........')
        DELP = np.array(dataset['DELP'][:][sel_time,:, sel_lat, sel_lon])
        print('Finished querying opendap server')

        #Save output
        met_dict = {'PS':PS/100, 'PL':PL/100, 'QV':QV, 'T':T, 'DELP':DELP/100}
        pickle.dump(met_dict, open('met/merra_met' + files.rad_name + '.p', 'wb'))

    else:
        metname = files.met_file
        met_dict = pickle.load(open(metname, 'rb'))
        PS = met_dict['PS']
        PL = met_dict['PL']
        QV = met_dict['QV']
        T = met_dict['T']
        DELP = met_dict['DELP']
        
    return PS, PL, QV, T, DELP


#Given a downloaded MERRA-2 meterological profile,
#organize met data and compute VCD, etc
def load_met(set_inputs, set_flight, files):

    met = make_class()
    
    #Load MERRA
    PS, PL, QV, T, DELP = load_merra(files, set_flight)

    def baro_func(P,T,PS): 
        Rstar = 0.0008352
        H = (Rstar * np.mean(T)) / (.0029 * 9.81)
        Z = -1 * (np.log(P/ PS)) * H
        return Z

    def scale_height(T):
        Rstar = 0.0008352
        return (Rstar * np.mean(T)) / (.0029 * 9.81)

    Z = baro_func(PL,T,PS)

    #Get pressure edges
    Pedge = [0.01]
    for idx in range(len(PL)):
        Pedge.append((PL[idx] + (DELP[idx]/2)))

    #Define default CH4 and N2O profiles
    low_alt = [120, 104.85, 55, 31.33, 20.68, 16.35, 11.91, 9.32,\
               7.35, 5.72, 4.33, 3.09, 2, 0.98, 0]
    low_ch4 = [3e-2, 1.103e-1, 1.65e-1, 8.692e-1, 1.51, 1.723, 1.823,\
               1.853, 1.862, 1.864, 1.864, 1.864, 1.864, 1.864, 1.864]
    low_n2o = [2.851e-4, 2.999e-3, 1.283e-1, 2.308e-1, 2.918e-1, 3.18e-1,\
               3.273e-1, 3.284e-1, 3.284e-1, 3.284e-1, 3.284e-1, 3.284e-1, 3.284e-1, 3.284e-1, 3.284e-1]

    ch4_lay = interpolate.interp1d(low_alt, low_ch4)(Z) / 1e6
    n2o_lay = interpolate.interp1d(low_alt, low_n2o)(Z) / 1e6

    #Add variables to met class
    met.Tmid = T
    met.zmid = Z
    met.pmid = PL
    
    #VMR variables
    met.vmr_CH4 = np.float64(ch4_lay)
    met.vmr_H2O = np.float64(QV * (18/28.9))
    met.vmr_N2O = np.float64(n2o_lay)
    
    #Compute Solar Zenith Angle
    dtime = datetime.datetime(int(str(set_flight.date_flight)[0:4]), \
                              int(str(set_flight.date_flight)[4:6]), \
                              int(str(set_flight.date_flight)[6:8]), \
                              set_flight.hour_utc, set_flight.min_utc, set_flight.sec_utc, \
                              tzinfo=datetime.timezone.utc)

    SZA = 90 - get_altitude(set_flight.latitude, set_flight.longitude, dtime)
    met.SZA = SZA
    
    #Set up reduced grid indices and define constants
    R_universal = 8.314472
    Na = 6.0221415e23
    go = 9.8196
    Rd = 287.04

    #Get values over midpoints of full grid
    dz = np.zeros(len(met.pmid))
    for idx in range(len(dz)):
        dz[idx] = np.log(Pedge[idx+1]/Pedge[idx])*Rd*T[idx]*(1+0.608*met.vmr_H2O[idx])/go

    rho_N =  PL*(1-met.vmr_H2O*1.6068)*100./(R_universal*T)*Na/10000.0

    #Add variables to met class
    met.dz = dz
    met.VCD_dry = np.float64(np.array(rho_N * dz))
    
    #Modify the number of gas layers in retrieval
    #-----For airbone, we want 2 - one above, one below
    #-----For satellite, we want 1 - everything below
    if set_flight.alt_agl_km < np.max(met.zmid):
 
        #Since we are an aircraft, set layers to 2
        set_inputs.layers_ch4 = 2
        set_inputs.layers_h2o = 2
        set_inputs.layers_n2o = 2
        
        #Set up pressure bounds
        where_is_aircraft = np.argmin(np.abs(set_flight.alt_agl_km - met.zmid))
        bnds = np.array([0, Pedge[where_is_aircraft], 1013])
        met.bnd_ch4 = bnds
        met.bnd_h2o = bnds
        met.bnd_n2o = bnds
        
        #Compute AMF
        AMF_below = (1./np.cos(math.radians(met.SZA))) + (1./np.cos(0))
        AMF_above = (1./np.cos(0))
        AMFS = np.zeros(len(met.pmid))
        AMFS[met.pmid < bnds[1]] = AMF_above
        AMFS[met.pmid >= bnds[1]] = AMF_below
        met.AMF = AMFS
        
        #make H's
        H_above = met.VCD_dry[met.pmid < bnds[1]] / np.sum(met.VCD_dry[met.pmid < bnds[1]])
        H_below = met.VCD_dry[met.pmid >= bnds[1]] / np.sum(met.VCD_dry[met.pmid >= bnds[1]])
        met.h_full = [H_above, H_below]
        

    else:
        
        #Since we are a satellite, set number of layers to one
        set_inputs.layers_ch4 = 1
        set_inputs.layers_h2o = 1
        set_inputs.layers_n2o = 1

        #Set up pressure bounds for each gas
        met.bnd_ch4 = np.linspace(0, 1013, set_inputs.layers_ch4+1)
        met.bnd_h2o = np.linspace(0, 1013, set_inputs.layers_h2o+1)
        met.bnd_n2o = np.linspace(0, 1013, set_inputs.layers_n2o+1)
        
        #Compute AMF
        AMF_below = (1./np.cos(math.radians(met.SZA))) + (1./np.cos(0))
        AMFS = [AMF_below] * len(met.pmid)
        met.AMF = np.array(AMFS)
        
        #Compute H
        met.h_full = np.float64(np.array([met.VCD_dry/np.sum(met.VCD_dry)]))
    
    #Air density in kg/m3
    ad = met.pmid / (Rd * met.Tmid * 1000)
    air_mole = ad * 1000 * (1/28.9) #density in moles/m3
    met.air_mole = air_mole
    met.ad = ad

    #Pressure weighting function
    NVAR = set_inputs.layers_ch4 + set_inputs.layers_h2o + set_inputs.layers_n2o + 1 + set_inputs.deg_poly + 1
    set_inputs.NVAR = NVAR
    h = np.zeros(set_inputs.layers_ch4)
    for idx in range(set_inputs.layers_ch4):
        isel = (met.pmid >= met.bnd_ch4[idx]) & (met.pmid < met.bnd_ch4[idx+1])
        h[idx] = np.sum(met.VCD_dry[isel])/np.sum(met.VCD_dry)
    met.h = h
    
    return met

#Do Gaussian convolution
@jit
def convolve_F(F, wvl_hi, FWHM):
    return gaussian_filter1d(F, FWHM / np.abs(wvl_hi[1]-wvl_hi[0]) / 2.355)

#Do sampling of hi-res to low res grid
@jit
def F_lo(F_hi, wvl_hi, wvl_lo, FWHM):
    return interpolate.interp1d(wvl_hi, convolve_F(F_hi, wvl_hi, FWHM))(wvl_lo)

#Either load Hitran absorption cross sections or download them using
#the pressure/temperature met profile
def load_hitran_solar(wave_class, met, files):

    #Download Hitran absorption cross sections using Hapi
    HWVL_max = 10000000 / wave_class.hitran_wvl[0]
    HWVL_min = 10000000 / wave_class.hitran_wvl[1]

    if files.hitran_loaded:

        cs_matrix_h2o = pickle.load( open( 'hitran/cs_matrix_h2o' + files.hitran_loaded + '.p', 'rb' ) )
        cs_matrix_ch4 = pickle.load( open( 'hitran/cs_matrix_ch4' + files.hitran_loaded + '.p', 'rb' ) )
        cs_matrix_n2o = pickle.load( open( 'hitran/cs_matrix_n2o' + files.hitran_loaded + '.p', 'rb' ) )
        nu_ = pickle.load( open( 'hitran/cs_matrix_nu' + files.hitran_loaded + '.p', 'rb' ) )


    else:

        #Load Hitran gas tables
        fetch('hitran/H2O' + files.rad_dir , 1, 1, HWVL_min, HWVL_max )
        fetch('hitran/CH4' + files.rad_dir, 6, 1, HWVL_min, HWVL_max )
        fetch('hitran/N2O' + files.rad_dir, 4, 1, HWVL_min, HWVL_max )


        #Use Hapi to compute cross sections at each level in the atmosphere
        for i in range(len(met.pmid)):
            p_ = met.pmid[i]/1013. 
            T_ = met.Tmid[i]      
            nu_, cs_h2o = absorptionCoefficient_Voigt(SourceTables='hitran/H2O' + files.rad_name, \
                                                      WavenumberRange=[HWVL_min,HWVL_max], \
                                                      Environment={'p':p_,'T':T_}, \
                                                      WavenumberStep=0.04)
            nu_, cs_ch4 = absorptionCoefficient_Voigt(SourceTables='hitran/CH4' + files.rad_name, \
                                                      WavenumberRange=[HWVL_min,HWVL_max], \
                                                      Environment={'p':p_,'T':T_}, \
                                                      WavenumberStep=0.04)
            nu_, cs_n2o = absorptionCoefficient_Voigt(SourceTables='hitran/N2O' + files.rad_name, \
                                                      WavenumberRange=[HWVL_min,HWVL_max], \
                                                      Environment={'p':p_,'T':T_}, \
                                                      WavenumberStep=0.04)
            if i == 0:
                cs_matrix_h2o = np.zeros((len(nu_), len(met.pmid)))
                cs_matrix_ch4 = np.zeros((len(nu_), len(met.pmid)))
                cs_matrix_n2o = np.zeros((len(nu_), len(met.pmid)))

            cs_matrix_h2o[:,i] = cs_h2o
            cs_matrix_ch4[:,i] = cs_ch4
            cs_matrix_n2o[:,i] = cs_n2o

            print(i, 'level cross sections done')

        pickle.dump(cs_matrix_h2o, open( 'hitran/cs_matrix_h2o' + files.rad_dir + '.p', 'wb' ) )
        pickle.dump(cs_matrix_ch4, open( 'hitran/cs_matrix_ch4' + files.rad_dir + '.p', 'wb' ) )
        pickle.dump(cs_matrix_n2o, open( 'hitran/cs_matrix_n2o' + files.rad_dir + '.p', 'wb' ) )
        pickle.dump(nu_, open( 'hitran/cs_matrix_nu' + files.rad_dir + '.p', 'wb' ) )

    #Keep cross-sections in a class
    cs = make_class()
    cs.cs_matrix_ch4 = cs_matrix_ch4
    cs.cs_matrix_h2o = cs_matrix_h2o
    cs.cs_matrix_n2o = cs_matrix_n2o

    #Get wavelength in nm and flip
    wvl_hi = 10000000 / nu_
    wave_class.wvl_hi = wvl_hi
    wave_class.nu_ = nu_

    #Load solar spectrum
    solar = np.loadtxt(files.solar_spec)
    Sfun = interpolate.interp1d(solar[:,0], solar[:,1])
    Ssolar = Sfun(wvl_hi)
    met.Ssolar = Ssolar
    
    return cs, met, wave_class


#Forward model - multiply transmission spectra by surface polynomial
@jit(nopython=True)
def Forward3(T, lval):
    return T * lval


#Compute Transmission spectra
#T_hi_res = I_0 * np.exp(-AMF * tau * scale_factor)
@jit
def Transmission(scaling_ch4, scaling_h2o, scaling_n2o, met, cs):

    #Initialize optical depth
    tau = np.zeros(cs.cs_matrix_ch4.shape[0])

    #Do methane optical depth
    for idx in range(len(met.bnd_ch4)-1):
        iscale = scaling_ch4[idx]
        isel = (met.pmid >= met.bnd_ch4[idx]) & (met.pmid < met.bnd_ch4[idx+1])
        new_layer_vmr = scaling_ch4[idx] * met.vmr_CH4[isel]
        layer_od = np.sum(met.AMF[isel] * cs.cs_matrix_ch4[:,isel] * new_layer_vmr * met.VCD_dry[isel],1)
        tau += layer_od

    #Do H2O optical depth
    for idx in range(len(met.bnd_h2o)-1):
        iscale = scaling_h2o[idx]
        isel = (met.pmid >= met.bnd_h2o[idx]) & (met.pmid < met.bnd_h2o[idx+1])
        new_layer_vmr = scaling_h2o[idx] * met.vmr_H2O[isel]
        layer_od = np.sum(met.AMF[isel] * cs.cs_matrix_h2o[:,isel] * new_layer_vmr * met.VCD_dry[isel],1)
        tau += layer_od

    #Do N2O optical depth
    for idx in range(len(met.bnd_n2o)-1):
        iscale = scaling_n2o[idx]
        isel = (met.pmid >= met.bnd_n2o[idx]) & (met.pmid < met.bnd_n2o[idx+1])
        new_layer_vmr = scaling_n2o[idx] * met.vmr_N2O[isel]
        layer_od = np.sum(met.AMF[isel] * cs.cs_matrix_n2o[:,isel] * new_layer_vmr * met.VCD_dry[isel],1)
        tau += layer_od

    #Compute Transmission spectrum
    #T_hi = met.Ssolar * np.exp(-tau)
    T_hi  = Forward3(met.Ssolar, np.exp(-tau))
    
    return T_hi




#Function to create layered VMR depending on number of atmospheric levels
@jit(nopython=True)
def vmr_red(vmr, bnds, pmid, h_full):
    mean_vmr = np.zeros(len(bnds)-1)
    for idx in range(len(bnds)-1):
        isel = (pmid >= bnds[idx]) & (pmid < bnds[idx+1])
        mean_vmr[idx] = np.dot(h_full[idx].T, vmr[isel])
    return mean_vmr

#Function to derive scale factors from reducted VMR profile
@jit(nopython=True)
def get_scaling(vmr, vmr_red, bnds, pmid, h_full):
    scaling = np.zeros(len(bnds)-1)
    for idx in range(len(bnds)-1):
        isel = (pmid >= bnds[idx]) & (pmid < bnds[idx+1])
        scaling[idx] = vmr_red[idx] / np.dot(h_full[idx].T, vmr[isel])
    return scaling


#Function to load observations and associated wavelength metadata
@jit
def load_raw_observations(files, set_inputs, wave_class, met, cs):

    #Load observations - BIL oe BSQ hyperspectral format

    #Load wavelength positions - either from file
    #or define range using FWHM
    wvl_file = np.sort(np.loadtxt(files.wave_pos))
    wvl_pos = wvl_file.copy()

    #Select only wavelengths that you wish to use in retrieval
    wvl_sel = (wvl_pos >= wave_class.inversion_wvl[0]) & (wvl_pos <= wave_class.inversion_wvl[1])
    wvl_lo = wvl_pos[wvl_sel]

    img = envi.open(files.rad_head, files.rad)
    index_r = np.argmin(np.abs(wvl_pos - 680))
    index_g = np.argmin(np.abs(wvl_pos - 530))
    index_b = np.argmin(np.abs(wvl_pos - 470))
    rgb = img[:,:,[index_r,index_g,index_b]]

    #Select subscene
    if set_inputs.ctmf_mask_include:
        hsel = range(coord_sub.include_x_left,coord_sub.include_x_right)
        vsel = range(coord_sub.include_y_top,coord_sub.include_y_bottom)
        hgrid, vgrid = np.meshgrid(hsel, vsel)
        rad = img.asarray()
        rad_sub = rad[hgrid, vgrid, :].copy()
        rad_sub2 = rad_sub.copy()
    else:
        hsel = range(rgb.shape[0])
        vsel = range(rgb.shape[1])
        rad_sub2 = img.asarray()
        
    #Create new class to store wavelength information
    wave_class.wvl_pos = wvl_pos
    wave_class.wvl_lo = wvl_lo
    wave_class.wvl_sel = wvl_sel
    
    #Low res absorption spectrum 
    cs_low = F_lo(cs.cs_matrix_ch4[:,len(met.pmid)-1], wave_class.wvl_hi, wave_class.wvl_lo, set_inputs.fwhm)
    wave_class.cs_low = cs_low
    
    #Add spectral shift information
    wave_class.wvl_loc = np.where(wvl_sel)[0]
    wave_class.disp_x = np.linspace(-1, 1, len(wave_class.wvl_loc))
    wave_class.shift_coef = np.polynomial.polynomial.polyfit(wave_class.disp_x, wave_class.wvl_lo, 0)
    wave_class.sval = np.polynomial.polynomial.polyval(wave_class.disp_x, wave_class.shift_coef)[0]
    
    #Add grid for legendre polynomial fit
    wave_class.leg_x_hi = np.linspace(-1, 1, len(wave_class.wvl_hi))
    wave_class.leg_x = interpolate.interp1d(wave_class.wvl_hi, wave_class.leg_x_hi)(wave_class.wvl_lo)

    
    #Compute transmission given prior gas inputs
    ch4_red_A = vmr_red(met.vmr_CH4, met.bnd_ch4, met.pmid, met.h_full)
    h2o_red_A = vmr_red(met.vmr_H2O, met.bnd_h2o, met.pmid, met.h_full)
    n2o_red_A = vmr_red(met.vmr_N2O, met.bnd_n2o, met.pmid, met.h_full)

    scaling_ch4_A = get_scaling(met.vmr_CH4, ch4_red_A, met.bnd_ch4, met.pmid, met.h_full)
    scaling_h2o_A = get_scaling(met.vmr_H2O, h2o_red_A, met.bnd_h2o, met.pmid, met.h_full)
    scaling_n2o_A = get_scaling(met.vmr_N2O, n2o_red_A, met.bnd_n2o, met.pmid, met.h_full)

    T_0 = Transmission(scaling_ch4_A, scaling_h2o_A, scaling_n2o_A, met, cs)
    T_lo_0 = F_lo(T_0, wave_class.wvl_hi, wave_class.wvl_lo, set_inputs.fwhm)
    
    met.ch4_red_A = ch4_red_A
    met.h2o_red_A = h2o_red_A
    met.n2o_red_A = n2o_red_A
    met.T_lo_0 = T_lo_0
    met.T_0 = T_0
    
    return rgb, rad_sub2, wave_class, met



def opt_depth(scaling_ch4, scaling_h2o, scaling_n2o, met, cs):

    #Initialize optical depth
    tau = np.zeros(cs.cs_matrix_ch4.shape[0])

    #Do methane optical depth
    for idx in range(len(met.bnd_ch4)-1):
        iscale = scaling_ch4[idx]
        isel = (met.pmid >= met.bnd_ch4[idx]) & (met.pmid < met.bnd_ch4[idx+1])
        new_layer_vmr = scaling_ch4[idx] * met.vmr_CH4[isel]
        layer_od = np.sum(met.AMF[isel] * cs.cs_matrix_ch4[:,isel] * new_layer_vmr * met.VCD_dry[isel],1)
        tau += layer_od

    #Do H2O optical depth
    for idx in range(len(met.bnd_h2o)-1):
        iscale = scaling_h2o[idx]
        isel = (met.pmid >= met.bnd_h2o[idx]) & (met.pmid < met.bnd_h2o[idx+1])
        new_layer_vmr = scaling_h2o[idx] * met.vmr_H2O[isel]
        layer_od = np.sum(met.AMF[isel] * cs.cs_matrix_h2o[:,isel] * new_layer_vmr * met.VCD_dry[isel],1)
        tau += layer_od

    #Do N2O optical depth
    for idx in range(len(met.bnd_n2o)-1):
        iscale = scaling_n2o[idx]
        isel = (met.pmid >= met.bnd_n2o[idx]) & (met.pmid < met.bnd_n2o[idx+1])
        new_layer_vmr = scaling_n2o[idx] * met.vmr_N2O[isel]
        layer_od = np.sum(met.AMF[isel] * cs.cs_matrix_n2o[:,isel] * new_layer_vmr * met.VCD_dry[isel],1)
        tau += layer_od

    #Compute Transmission spectrum
    #T_hi = met.Ssolar * np.exp(-tau)
    
    return tau


def Make_Jac4(T, lcoefs, scoefs, wave_class, met, set_inputs, cs, sCH4, sH2O, sN2O):

    #Redefine wavelength grid
    hi_x = np.linspace(-1, 1, len(wave_class.wvl_hi))
    lo_x = interpolate.interp1d(np.flip(wave_class.wvl_hi), hi_x)(wave_class.wvl_lo)
    
    #Run baseline forward model
    K = np.zeros((len(wave_class.wvl_lo), set_inputs.NVAR))
    lval = np.polynomial.polynomial.polyval(hi_x, lcoefs)
    sval = np.polynomial.polynomial.polyval(lo_x, scoefs)[0]
    F = Forward3(T, lval)
    Flo = F_lo(F, np.flip(wave_class.wvl_hi), wave_class.wvl_lo, set_inputs.fwhm)
    tot_od = np.flip(opt_depth(sCH4, sH2O, sN2O, met, cs))
    exp_od = np.exp(-tot_od)

    #CH4 Jacobian
    ii = 0
    for idx in range(set_inputs.layers_ch4):
        isel = (met.pmid >= met.bnd_ch4[idx]) & (met.pmid < met.bnd_ch4[idx+1])
        pre_fac = np.flip(met.Ssolar) * lval * \
                np.sum(met.AMF[isel] * np.flip(cs.cs_matrix_ch4[:,isel], axis=0) * \
                met.vmr_CH4[isel] * met.VCD_dry[isel],1)
        K_hi = -pre_fac * exp_od
        K[:,ii] = F_lo(K_hi, np.flip(wave_class.wvl_hi), wave_class.wvl_lo, set_inputs.fwhm)
        ii += 1

    #H2O Jacobian
    for idx in range(set_inputs.layers_ch4):
        isel = (met.pmid >= met.bnd_ch4[idx]) & (met.pmid < met.bnd_ch4[idx+1])
        pre_fac = np.flip(met.Ssolar) * lval * \
                np.sum(met.AMF[isel] * np.flip(cs.cs_matrix_h2o[:,isel], axis=0) * \
                met.vmr_H2O[isel] * met.VCD_dry[isel],1)
        K_hi = -pre_fac * exp_od
        K[:,ii] = F_lo(K_hi, np.flip(wave_class.wvl_hi), wave_class.wvl_lo, set_inputs.fwhm)
        ii += 1

    #N2O Jacobian
    for idx in range(set_inputs.layers_ch4):
        isel = (met.pmid >= met.bnd_ch4[idx]) & (met.pmid < met.bnd_ch4[idx+1])
        pre_fac = np.flip(met.Ssolar) * lval * \
                np.sum(met.AMF[isel] * np.flip(cs.cs_matrix_n2o[:,isel], axis=0) * \
                met.vmr_N2O[isel] * met.VCD_dry[isel],1)
        K_hi = -pre_fac * exp_od
        K[:,ii] = F_lo(K_hi, np.flip(wave_class.wvl_hi), wave_class.wvl_lo, set_inputs.fwhm)
        ii += 1

    #Jacobian with respect to spectral shift
    dDispersion = scoefs * 1e-3 
    for idx in range(len(dDispersion)):
        new_poly = scoefs.copy()
        new_poly[idx] += dDispersion[idx]
        new_wvl = wave_class.wvl_lo * \
            (np.polynomial.polynomial.polyval(lo_x, new_poly)[0] / wave_class.sval)
        Fp = F_lo(F, np.flip(wave_class.wvl_hi), new_wvl, set_inputs.fwhm)
        K[:, ii] = (Fp - Flo) / dDispersion[idx]
        ii += 1

    #Jacobian with respect to Legendre polynomial coefficients
    for idx in range(len(lcoefs)):
        K_hi = np.flip(met.Ssolar) * np.exp(-tot_od) * (hi_x**idx)
        K[:, ii] = F_lo(K_hi, np.flip(wave_class.wvl_hi), wave_class.wvl_lo, set_inputs.fwhm)
        ii += 1

    return K


def retrieve_scene2(Y, set_inputs, wave_class, met, cs):
    
    inv_prms = make_class()

    #Prior start of polynomial fit is fit to subsample of Y
    init_k = set_inputs.deg_poly
    YT = Y/met.T_lo_0

    hi_x = np.linspace(-1, 1, len(wave_class.wvl_hi))
    lo_x = interpolate.interp1d(np.flip(wave_class.wvl_hi), hi_x)(wave_class.wvl_lo)

    if set_inputs.do_legendre:
        near_zeros = np.where(wave_class.cs_low < np.percentile(wave_class.cs_low, 40))[0]
        lfit = np.polynomial.legendre.legfit(lo_x[near_zeros], YT[near_zeros], init_k)
        lval = np.polynomial.legendre.legval(hi_x, lfit)
        lval_lo = np.polynomial.legendre.legval(lo_x, lfit)

    else:
        near_zeros = np.where(wave_class.cs_low < np.percentile(wave_class.cs_low, 40))[0]
        lfit = np.polynomial.polynomial.polyfit(lo_x[near_zeros], Y[near_zeros], init_k)
        lval = np.polynomial.polynomial.polyval(hi_x, lfit)
        lval_lo = np.polynomial.polynomial.polyval(lo_x, lfit)

    #Inverse via iterative solution
    lCH4 = set_inputs.layers_ch4
    lH2O = set_inputs.layers_h2o
    lN2O = set_inputs.layers_n2o

    #Define first prior
    s1 = get_scaling(met.vmr_CH4, met.ch4_red_A, met.bnd_ch4, met.pmid, met.h_full)
    s2 = get_scaling(met.vmr_H2O, met.h2o_red_A, met.bnd_h2o, met.pmid, met.h_full)
    s3 = get_scaling(met.vmr_N2O, met.n2o_red_A, met.bnd_n2o, met.pmid, met.h_full)

    #Set value of first prior + first iteration
    xa_full = np.append(s1, np.append(s2, np.append(s3, np.append(wave_class.shift_coef, lfit))))
    inv_prms.xa_full = xa_full
    ix = xa_full.copy()

    #Construct prior error covariance
    sig_prior = np.abs(ix.copy())  #Error on prior proportional to magnitude

    #Methane prior
    sig_prior[0] *= 2
    if set_inputs.layers_ch4 == 2: #If airbone
        sig_prior[1] *= 2    
    sig_prior[(lCH4):(lCH4+lH2O)] *= 1e-1 #H2O prior
    sig_prior[(lCH4+lH2O):(lCH4+lH2O+lN2O)] *= 1e-1 #N2O prior
    sig_prior[lCH4+lH2O+lN2O] *= 1e-5 #Shift prior
    sig_prior[-len(lfit):] *= 1e-1 #polynomial
    Sa = np.diag((sig_prior))
    invSa = np.linalg.inv(Sa)

    #Obs error covariance
    noise_val = 1/(set_inputs.SNR**2)
    invSe = np.diag([1/noise_val] * len(Y))

    #Do iterative solution
    num_iter = 1
    x_out = np.zeros((len(xa_full), num_iter))

    i_iter = 0

    #Run Forward model
    ilval = np.polynomial.polynomial.polyval(hi_x, ix[-len(lfit):])
    T = np.flip(Transmission(ix[0:(lCH4)], ix[(lCH4):(lCH4+lH2O)], ix[(lCH4+lH2O):(lCH4+lH2O+lN2O)], met, cs))
    Fa = F_lo(Forward3(T, ilval), np.flip(wave_class.wvl_hi), wave_class.wvl_lo, set_inputs.fwhm)

    #Create Jacobian
    K = Make_Jac4(T, lcoefs=ix[-len(lfit):], scoefs=np.array([ix[lCH4+lH2O+lN2O]]), \
            wave_class=wave_class, met=met, set_inputs=set_inputs, cs=cs, \
            sCH4=ix[0:(lCH4)], sH2O=ix[(lCH4):(lCH4+lH2O)], sN2O=ix[(lCH4+lH2O):(lCH4+lH2O+lN2O)])

    #Iterative solution for next state
    term1 = np.linalg.inv(K.T.dot(invSe).dot(K) + invSa)
    term2 = K.T.dot(invSe)
    term3 = (Y - Fa) + K.dot(ix - xa_full)
    x_1 = xa_full + (term1.dot(term2)).dot(term3)

    #term1 = np.linalg.inv(np.dot(K.T, np.dot(invSe, K)) + invSa)
    #term2 = np.dot(K.T, invSe)
    #term3 = Fa - Y
    #x_1 = xa_full + np.dot(term1, np.dot(term2, term3))
    
    #Posterior error
    S_1 = np.linalg.inv(K.T.dot(invSe).dot(K)+np.linalg.inv(Sa))
    lval_hat = np.polynomial.polynomial.polyval(hi_x, x_1[-len(lfit):])

    #Posterior estimate
    sCH4_hat = x_1[0:lCH4]
    sH2O_hat = x_1[(lCH4):(lCH4+lH2O)]
    sN2O_hat = x_1[(lCH4+lH2O):(lCH4+lH2O+lN2O)]

    That = np.flip(Transmission(sCH4_hat, sH2O_hat, sN2O_hat, met, cs))
    Fhat = F_lo(Forward3(That, lval_hat), np.flip(wave_class.wvl_hi), wave_class.wvl_lo, set_inputs.fwhm)

    #Output data
    inv_prms.Fh = Fhat
    inv_prms.Y = Y
    inv_prms.Fa = Fa
    inv_prms.RMSE = np.sqrt(np.mean((Y-Fhat)**2))
    inv_prms.ilval = lval_lo
    inv_prms.lval_hat = F_lo(lval_hat, np.flip(wave_class.wvl_hi), wave_class.wvl_lo, set_inputs.fwhm)
    inv_prms.K = K
    inv_prms.xhat = x_1
    inv_prms.Sa = Sa
    inv_prms.sCH4_hat = sCH4_hat
    inv_prms.sH2O_hat = sH2O_hat
    inv_prms.sN2O_hat = sN2O_hat
    inv_prms.Shat = S_1

    xa_diag = np.diag(xa_full)
    inv_prms.Shat_conv = (xa_diag.dot(S_1)).dot(xa_diag.T)

    retrieved_CO2 = (np.dot(x_1[0:lCH4] * met.ch4_red_A, met.h) * 1e9) / 1000
    #retrieved_CO2 = x_1[0] * met.ch4_red_A[0] * 1e6


    return inv_prms, retrieved_CO2


#Function to run retrieval over an entire scene
def run_retrieval(rad_sub_enmap, wave_class, set_inputs, met, cs, scaling = 1, out_name='temp'):

    #Write profile data
    prof_dict = {'VCD': met.VCD_dry, 'VMR_CH4': met.vmr_CH4, 'VMR_H2O': met.vmr_H2O,  \
                 'Tmid': met.Tmid, 'Pmid': met.pmid, 'Zmid':met.zmid}
    prof_df = pd.DataFrame.from_dict(prof_dict, orient='columns')
    prof_df.to_csv('output/profile_info.csv')

    #Run retrieval
    try:
        xCH4_ENMAP = pickle.load(open('./output/' + out_name + '.p', 'rb'))
    except:

        xCH4_ENMAP = np.zeros(rad_sub_enmap.shape[0:2])
        SF = np.zeros(rad_sub_enmap.shape[0:2])
        SH = np.zeros(rad_sub_enmap.shape[0:2])

    xsel, ysel = np.where(xCH4_ENMAP == 0)

    print_array = range(0, xCH4_ENMAP.shape[0], 10)
    out_prms = {}
    for idx in range(len(xsel)):
            isel = xsel[idx]
            jsel = ysel[idx]
            Y = rad_sub_enmap[isel, jsel, wave_class.wvl_sel] / scaling
            inv_prms, retrieved_CH4 = retrieve_scene2(Y, set_inputs, wave_class, met, cs)

            xCH4_ENMAP[isel,jsel] = retrieved_CH4
            SF[isel,jsel] = inv_prms.sCH4_hat[0]
            SH[isel, jsel] = inv_prms.Shat[0,0]
            
            out_prms['pixel_'+ str(isel) + str(jsel)] = inv_prms
            
            pickle.dump(xCH4_ENMAP, open('./output/' + out_name + '.p', 'wb'))
            pickle.dump(SF, open('./output/' + out_name + '_SF.p', 'wb'))
            pickle.dump(SH, open('./output/' + out_name + '_SH.p', 'wb'))
            
    return xCH4_ENMAP, out_prms





















#!/usr/bin/env python

from imap_functions import *


#Load command line arguments
import argparse

parser = argparse.ArgumentParser(description='Arguments for CO2 IMAP-DOAS model')
parser.add_argument('-d','--direc', help='Directory when BIP file is located',required=True)
parser.add_argument('-n','--name', help='file name',required=True)
parser.add_argument('-s','--source', help='Plume source ID',required=True)
parser.add_argument('-x','--lon', help='Plume longitude',required=True)
parser.add_argument('-y','--lat', help='Plume latitude',required=True)
parser.add_argument('-m','--met', help='Name of met file',required=True)
parser.add_argument('-t','--hitran', help='Name of hitran file',required=True)

args = parser.parse_args()


#Set up the retrieval
init_dict = {'rad_dir': args.direc,\
             'rad_name': args.name,\
             'main': '/Users/cusworth/Documents/IMAP_DOAS/input/',\
             'wave_pos': 'ancillary/av_wvl.txt',\
             'alt_agl_km': 3,\
             'latitude': float(args.lat),\
             'longitude': -1*float(args.lon),\
             'fwhm': 5,\
             'SNR': 200,\
             'deg_poly': 8,\
             'do_legendre': False, \
             'met_file':'met/merra_met'+ args.met + '.p',\
             'inversion_wvl': [2215, 2415], \
             'hitran_loaded': args.hitran}


#Load image and data
files, set_inputs, wave_class, set_flight, coord_sub = initialize(init_dict)
met = load_met(set_inputs, set_flight, files)
cs, met, wave_class = load_hitran_solar(wave_class, met, files)
rgb, rad_sub, wave_class, met = load_raw_observations(files, set_inputs, wave_class, met, cs)


#Run Retrieval
xCO2, out_prms = run_retrieval(rad_sub, wave_class, set_inputs, met, cs, 10, args.source)




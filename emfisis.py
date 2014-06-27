from spacepy import pycdf
import spacepy
import numpy as np
import os
import datetime
from scipy.interpolate import interp1d
from spacepy import datamodel as dm

EMFISISDIR='/n/toaster/u/jhaiduce/data/EMFISIS'

def get_filename(scname,date,version='1.3.10'):
    """
    Determines the filename of the CDF file corresponding to the requested spacecraft and date
    """

    # Set spacecraft name to match emfisis filename convention
    if scname.lower() in('rbspa','rbsp-a'):
        scname='rbsp-a'
    if scname.lower() in('rbspb','rbsp-b'):
        scname='rbsp-b'

    filename='{scname:s}_density_emfisis_{year:d}{month:02d}{day:02d}_v{version:s}.cdf'.format(scname=scname.lower(),year=date.year,month=date.month,day=date.day,version=version)
    return os.path.join(EMFISISDIR,filename)

def get_data(scname,varnames,dstart,dend):
    """
    Gets EMFISIS data
    """

    ordinal_dates=range(dstart.toordinal(),dend.toordinal())
    
    try:
        _ = (varname for varname in varnames)
    except TypeError:
        varnames=[varnames]

    values=[[] for _ in xrange(len(varnames))]

    for ordinal in ordinal_dates:
        date=datetime.datetime.fromordinal(ordinal)
        try:
            filename=get_filename(scname,date)
            cdffile=dm.fromCDF(filename)
        except spacepy.pycdf.CDFError:
            filename=get_filename(scname,date,version='1.3.8')
            cdffile=dm.fromCDF(filename)
            
        for i,varname in enumerate(varnames):
            cdfvar=cdffile[varname]
            arr=np.ma.masked_equal(np.array(cdfvar),cdfvar.attrs['FILLVAL'])
            values[i].append(arr)

    for i in range(len(varnames)):
        values[i]=np.ma.concatenate(values[i])

    return values

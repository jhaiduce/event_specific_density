"""
The emfisis module contains tools for reading density data collected by the Themis satellites.

.. rubric:: Functions

.. autosummary::
    :toctree: autosummary
    
    load

"""
import numpy as np
import os
import datetime

THEMISDIR='/n/toaster/u/jhaiduce/data/themis'

def parsedate(s):
    return datetime.datetime.strptime(s,'%Y-%m-%d/%H:%M:%S')

def load():
    """
    Load all the Themis data into a single Numpy array
    """
    files=os.listdir(THEMISDIR)

    # List to hold the records before concatenation
    data=[]

    for filename in files:
        if not filename.endswith('.dat'):
            continue
        
        # Read this file
        thisdata=np.genfromtxt(os.path.join(THEMISDIR,filename),
                               dtype=[('time',object),('density',float),('in_plasmasphere',int),
                                      ('L',float),('MLT',float),('scnumber',int)],
                               converters={'time':parsedate})
        data.append(thisdata)

    # Concatenate the data into a single numpy array
    data=np.concatenate(data)
    return data

if __name__=='__main__':
    themis_data=load()
    print themis_data['scnumber']
    print themis_data['scnumber']==1

.. EMFISIS-fit Density Model documentation master file, created by
   sphinx-quickstart on Mon Jul  7 13:43:20 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EMFISIS-fit Density models module
=================================

This module contains the EMFISIS-fit plasma density model as well as several other density models which have been used in constructing the EMFISIS-fit model, and utilities for loading data required to use the models.

EMFISIS-fit Density Model
-------------------------

The EMFISIS-fit density model (:class:`densitymodels.emfisis_fit_model`) combines the Sheeley density model with the latitude dependence factor from the Ozhogin density model and a modified O'Brien and Moldwin (2003) plasmapause model. The plasmapause is handled by using a smooth polynomial interpolation function known as smoothstep (:func:`densitymodels.smoothstep`) to interpolate interpolate between the Sheeley plasmasphere density model and the Sheeley trough density model. 

This density model is fit on a least-squares basis to density data from one of the RBSP spacecraft. The orbital path of the spacecraft is broken up into half-orbits (that is, segements extending from one apogee to the next perigee, or from one perigee to the next apogee). 

EMFISIS-smoothing density model
-------------------------------

The EMFISIS-smoothing density model (:class:`densitymodles.emfisis_smoothing_model`) interpolates and optionally smooths the denisty data from orbits of the RBSP spacecraft as a function of radius from the Earth, and applies the resulting densities to all local times. Latitude dependence is added obtained using the latitude dependence factor of the Ozhogin model. 

Supporting modules and classes
------------------------------

The :mod:`ephemeris` module contains functions for reading and interpolating ephemeris data from LANL MagEphem files.

The :mod:`emfisis` module contains functions for reading density data obtained using the EMFISIS instrument.

The :mod:`themisdata` module contains functions for reading density data obtained from the Themis satellites.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


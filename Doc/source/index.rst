.. EMFISIS-fit Density Model documentation master file, created by
   sphinx-quickstart on Mon Jul  7 13:43:20 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EMFISIS-fit Density Model
=====================================================

The EMFISIS-fit density model combines the Sheeley density model with the latitude dependence factor from the Ozhogin density model. The plasmapause is handled by using a smooth polynomial interpolation function known as smoothstep to interpolate interpolate between the Sheeley plasmasphere density model and the Sheeley trough density model. 

This density model is fit on a least-squares basis to density data from one of the RBSP spacecraft. The orbital path of the spacecraft is broken up into half-orbits (that is, segements extending from one apogee to the next perigee, or from one perigee to the next apogee). 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


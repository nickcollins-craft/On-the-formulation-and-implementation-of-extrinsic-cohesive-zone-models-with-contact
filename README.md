# On-the-formulation-and-implementation-of-extrinsic-cohesive-zone-models-with-contact
This repository stores the codes associated with the paper "On the formulation and implementation of extrinsic cohesive zone models with contact", by N.A. Collins-Craft, F. Bourrier and V. Acary, published in Computer Methods in Applied Mechanics and Engineering: https://doi.org/10.1016/j.cma.2022.115545

The codes are also available on Zenodo: https://doi.org/10.5281/zenodo.6939392

In order to run the codes, the user first requires a working installation of Python 3 (the codes were run on Python 3.8.13). We strongly recommend using Anaconda to manage the installation: https://www.anaconda.com/products/distribution
Then, the user must install meshio (the codes were run on meshio 5.3.0): https://github.com/nschloe/meshio
Separately, the user must also install Gmsh (the codes were run on Gmsh 4.8.1): https://gmsh.info/
Finally, the user must also install Siconos (the codes were run on Siconos 4.4.0): https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/

Then, those codes that call the Siconos software can be run by typing "siconos <name_of_code.py>" and pressing enter while in the working directory. All other codes can be run by typing "python <name_of_code.py>" and pressing enter while in the working directory. The user will need to modify the files by specifying the appropriate path to the folder that saves the data or figure.

Users of this software should cite the associated paper, and also each software listed above where appropriate.

Two of the codes ("DCB_mesh_and_end_plot.py" and "Rhombus_mesh_and_end_plot.py") also require images that are included in the related repository containing the output data: https://doi.org/10.5281/zenodo.6939154

Contact: nicholas[dot]collins-craft[at]inria[dot]fr

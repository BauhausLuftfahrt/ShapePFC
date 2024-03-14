# ShapePFC

**S**wift **H**ybrid numerical method for **A**ero-propulsive **P**erformance **E**valuation of **P**ropulsive 
**F**uselage **C**oncepts

The code in the repository was employed to study the aero-propulsive performance of Propulsive Fuselage Concepts (PFCs)
in aircraft conceptual design. For this reason, a hybrid numerical method was developed, validated, and applied to a 
design parameter study. It combines a viscous/inviscid panel method with a finite volume calculation of the RANS 
equations.

The following methods are implemented, some methods have not been employed in the final studies, but are included in 
this repository as an example for their implementation in Python.

1. Hybrid Panel/Finite Volume Method (HPFVM) for Propulsive Fuselage Concepts as documented in [0]. It includes
   1. `sample_generation` - Scripts to generate samples for a sensitivity study and for a multiparameter study 
   based on a Maximin Latin Hypercube Sampling (LHS) design of experiment
   2. `geometry_generation` - An intuitive Class/Shape Transformation (iCST) function method to generate 2D-axisymmetric 
   fuselage-propulsor geometries
   3. `panel` - The implementation of a viscous/invsicid panel method for axisymmetric streamlined bodies of revolution.
   It includes a potential flow solver (implementation of sources, doublets, and vortices) and an Integral Boundary 
   Layer (IBL) method [1, 2]. For the prediction of the turbulent boundary layer, two different methods are implemented,
   one by Green and one by Patel. The latter is used in the final study. The panel method can be run on its own. 
   Examples are given in `studies/panel_studies`
   4. `interface` - A method to transfer the solution of the panel method on the finite volume computation.
   5. `finite_volume` - Helping functions to set up up the finite volume computation implemented in OpenFOAM (v2206)
   6. `hybrid_method` - Exemplary OpenFOAM cases, which are used for the HPFVM studies
   7. `post_processing` - Functions to postprocess the results of the different solvers
   8. `studies` - This folder contains scripts to automate the studies, which were run as part of the project. The 
   scripts can be used as examples to set up other studies. The most important one is `studies/hybrid_fv_pm_studies/
   sensitivity_study/sensitivity_study.py`

2. A Finite Difference (FD) implementation for streamlined 2D-axisymmetric bodies (e.g., fuselages) and for 2D-planar 
bodies(e.g., nacelles) (`finite_differences`). A body- (or boundary-) fitted curvilinear coordinate systems is generated 
by the solution elliptic Partial Differential Equation (PDE) system. Until now, only the solution of the non-linear 
potential equation on the grid is implemented (but not thoroughly tested).
3. Hall-Thollet body force model implementation (`body_force_model`), which was intended to be used with the FD solver.


# Running the Hybrid Finite Volume/Panel Method Studies
Running the HPFVM studies as they are requires the following:

1. Replace the functions from the bhlpythontoolbox (i.e., the atmosphere model) by your own atmosphere model
2. Set up your own OpenFOAM environment, including the HiSA solver and the body force model. The studies ran successfully 
on OpenFOAM-v2206 and the corresponding HiSA version: [https://gitlab.com/hisa/hisa](https://gitlab.com/hisa/hisa). I used a setup on an Ubuntu OS.
3. Install gmsh on same machine (Gmsh v4.9.5 was used originally)
4. Install 


# Literature

The HPFVM code, its validation, application, assumptions and limitations is documented in 

    [0] Habermann, A.L.: Numerical Assessment of Propulsive Fuselage Performance in
        Aircraft Conceptual Design. Dissertation. Technical University Munich. To be published.

Further details on the implementation of the panel method can be found in

    [1] Santa Cruz Mendoza, C.E.R.: Numerical Prediction of Turbulent Boundary Layer Characteristics to Feed
        the Design of Propulsive Fuselages. Master’s thesis. University of Stuttgart. 2020.
    [2] Romanow, N.: Extension of a Numerical Model for the Prediction of Turbulent Boundary Layer Characteristics
        on a Propulsive Fuselage Aircraft. Master’s thesis. Technical University Munich. 2021.

Additional important sources are documented in the relevant functions and in [0].

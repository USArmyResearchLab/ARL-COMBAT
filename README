ARL Computational Model Builder and Analysis Toolkit (ARL-COMBAT)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The ARL Computational Model Builder and Analysis Toolkit (ARL COMBAT) is a collection of Python modules that can be used to create model structures and characterize their structural and material properties.
The ARL Computational Model Builder and Analysis Toolkit (ARL COMBAT) leverages functionality present within the LAMMPS software package, and can be used to analyze LAMMPS-formatted dump files.
The software was developed at the US Army CCDC Army Research Laboratory and partially funded through the DOD High Performance Computing Modernization Program (HPCMP) Internship Program.
Portions of the ARL COMBAT is distributed under the Creative Commons Zero 1.0 Universal (CC0 1.0) Public Domain Dedication and GNU General Public License v3.0.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The ARL Computational Model Builder and Analysis Toolkit (ARL COMBAT) distribution includes the following files and directories:
README 	   		this file
LICENSE              	the Creative Commons Zero 1.0 Universal (CC0 1.0) Public Domain Dedication and the GNU General Public License version 3.0
examples             	simple test problems
combat                	ARL COMBAT source files

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
PREREQUISITES

The following prerequisite software and libraries are required to be installed prior to using the ARL Computational Model Builder and Analysis Toolkit (ARL COMBAT):

1.  combat.py 		     located within the src/ directory
2.  nanocrystal_builder.py   located within the src/ directory
3.  LAMMPS 		     http://lammps.sandia.gov
4.  Python 		     https://www.python.org/
5.  NumPy 		     http://www.numpy.org/
6.  SciPy 		     https://www.scipy.org/
7.  pandas 		     http://pandas.pydata.org/
8.  mpi4py 		     http://mpi4py.scipy.org/docs/
9.  networkx 		     http://networkx.github.io/

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BUILD INSTRUCTIONS

The ARL Materials Modeling for Particle Models Software Suite leverages LAMMPS as well as the NumPy, SciPy, pandas and mpi4py Python modules.  
The Python modules can be installed by following the instructions provided on their respective websites.
Instructions for building LAMMPS are provided at http://lammps.sandia.gov.  
The following LAMMPS add-on packages are integral to the ARL Computational Model Builder and Analysis Toolkit (ARL COMBAT), and must be installed to run the examples located within the examples/ directory:

1.  MC:  required for polymer bond creation
2.  MISC:  required for monomer insertion
3.  MOLECULE:  required for polymer growth
4.  RIGID:  required for rigid expansion of polycrystalline materials
5.  VORONOI:  required for void volume calculations in crystalline materials

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
EXAMPLES

The following examples for running the software are provided within the examples/ directory:

1.  01_BuildSingleCrystal      -- Building an FCC single crystal (LAMMPS)
2.  02_BuildPolyCrystal        -- Building a polycrystalline sample (nanocrystal_builder.py)
3.  03_BuildComposite          -- Growing polymer chains in a polycrystalline sample (LAMMPS, combat.py)
4.  04_DetectVoids             -- Detect voids present within a crystal (LAMMPS, combat.py)
5.  05_IdentifyGBs             -- Identifying grain boundaries in a polycrystalline sample (combat.py)
6.  06_ComputeLocalTemperature -- Computing locally weighted averages of temperature in a single crystal (combat.py)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DOCUMENTATION

Documentation of the examples are provided in the ARL Technical Report entitled, "Pre- and Post-Processing Tools to Create and Characterize Particle-Based Composite Model Structures", ARL-TR-8213.
Documentation of the polycrystal builder is provided in the ARL Technical Note entitled, "Voronoi Based Nanocrystalline Generation Algorithm for Atomistic Simulations", ARL-TN-0806.
Readers are directed to the LAMMPS website at http://lammps.sandia.gov for more thorough documentation on the LAMMPS framework and user commands.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CITATIONS

If you are using this work in a publication, please include the following references in your paper:
Fortunato, M.E.; Mattson, J.; Taylor, D.E.; Larentzos, J.P.; Brennan, J.K.  "Pre- and Post-Processing Tools to Create and Characterize Particle-Based Composite Model Structures", ARL Technical Report, ARL-TR-8213, 2017.
Foley, D.; Coleman, S.P.; Tucker, G.; Tschopp, M.A.  "Voronoi Based Nanocrystalline Generation Algorithm for Atomistic Simulations", ARL Technical Note. ARL-TN-0806, 2016.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CONTRIBUTING AUTHORS:

Michael E. Fortunato, Department of Chemistry, University of Florida, Gainesville, FL.
Daniel Foley, Mechanical Engineering, Colorado School of Mines, Golden, CO.
Garritt Tucker, Mechanical Engineering, Colorado School of Mines, Golden, CO.
Joseph Mattson, Science and Math Academy, Aberdeen High School, Aberdeen, MD.
DeCarlos E. Taylor, Weapons and Materials Research Directorate, US Army CCDC Army Research Laboratory, Aberdeen Proving Ground, MD.
Shawn P. Coleman, Weapons and Materials Research Directorate, US Army CCDC Army Research Laboratory, Aberdeen Proving Ground, MD.
James P. Larentzos, Weapons and Materials Research Directorate, US Army CCDC Army Research Laboratory, Aberdeen Proving Ground, MD.
John K. Brennan, Weapons and Materials Research Directorate, US Army CCDC Army Research Laboratory, Aberdeen Proving Ground, MD.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

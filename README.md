# Steve_Marc_Raj
Tool to produce the histograms, which will fed to Marc's fitting tool 


This is a simple rdf code.

Main code - Steve.py

Steve.h contains the functions which are needed to create the TP pairs

To run - use the singularity shell of Josh - "singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest"

then "python Steve.py -h"

There is a file runAll.py, which is used to runn all the efficiency steps in one go. So do "python runAll.py -h" to see all the options.
All the options are self-explaining. However the important parameter is -r. The options are again self satisfctory. But here "stand" means the 
code will run on data and mc. "bkg" maens that the code will run on only the background processes. And "all" means code will run on the all 
processes.

## Synopsis

This is some python code to attempt to duplicate the calculations done in a paper about the EM Drive that was published by NASA's Eagleworks labratory.  You can find a PDF copy of this paper in this git hub archive.  The code attempts to duplicate their calcuations and estimates.  This was done without access to their raw data which required estimates and curve fitting.

## Background

See the files in this repository for more background information on how the plots were generated and how you can contribute to making this simulation either demonstrate the idea of superimposed impluse force from the EM Drive is properly analyzed.  It is my hope that the raw test data will be released to the public so it can be statistically analysed instead of simulated as it is done here.

![Eagleworks Paper in PDF](./final-paper.pdf)

![Eagleworks Paper with critical comments in PDF](./final-paper-comments.pdf)


## Motivation

The goal is to use this code to build methods to test the predictability of their impulse model and compare the simulation to their results.  This model can be made more complex by adding in additional sweeps and monte carlo simulations of different forces and shapes to compare them to what was predicted in the Eagleworks paper.

![Background explanation of code in PDF](./background.pdf)

![Some supporting calculations in libreoffice calc](./EW-data.ods)

## Installation

To run this simulation you need python 2.7+ or 3 and libraries: scipy, matplotlib and numpy

python test1.py
Runs the test with all Eagleworks time windows and best estimations (note an error in pulse force calculation exits)

python test2.py
An example that runs the same tests as test1.py with a minor adjustment on the force pulse window that fits the curve better. (note an error in pulse force calculation exits)

python test3.py
Has fix for force calculation and has a 6th order polynomial fit for thermal curve.  Also force pulse is set to 0 to try to duplicate the 106uN measurement.  Details about this are in both the [background.pdf](./background.pdf) and the [spreadsheet](./EW-data.ods).

python test4.py
This has a new dataset that was extracted digitally from Figure 7 of the Eagleworks paper.  The data set can be found in [ew-graph.csv](./ew-graph.csv)

## Contributors

Feel free to add your own models, curvfits or other data to this repository.

## License

Placed in the Public Domain.

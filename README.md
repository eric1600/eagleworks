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

## Test Overview

To run this simulation you need python 2.7+ or 3 and libraries: scipy, matplotlib and numpy

* test1.py (contains errors)
Runs the test with all Eagleworks time windows and best estimations (note an error in pulse force calculation exits)

* test2.py (contains errors)
An example that runs the same tests as test1.py with a minor adjustment on the force pulse window that fits the curve better. (note an error in pulse force calculation exits)

* test3.py (contains errors)
Has fix for force calculation and has a 6th order polynomial fit for thermal curve.  Also force pulse is set to 0 to try to duplicate the 106uN measurement.  Details about this are in both the [background.pdf](./background.pdf) and the [spreadsheet](./EW-data.ods).

* test4.py (matches EW calcuations)
This has a new dataset that was extracted digitally from Figure 7 of the Eagleworks paper.  The data set can be found in [ew-graph.csv](./ew-graph.csv)

* test5.py
This code tests out the models presented in Fig. 5 of the Eagleworks paper.  Curve fits were made for both the force pulse and the thermal profile.  The code was then written to scale both the time base and amplitude to allow for duplicating measured results just testing their model.  These results show that even with 0 force there is ~92 uN of force present using the peak scale of Figure 8.  This implies that they are unable to seperate the force from the thermal as they theorized.  More testing to be done.  One side note is the time window for the pulse calculation had to be shorted by a few seconds from Eagleworks numbers to fit the slope of their model better.  Click this link to ![see the graphs and short discussion of the test5 results](https://imgur.com/a/Whfiu).

## Contributors

Feel free to add your own models, curvfits or other data to this repository.

## License

Placed in the Public Domain.

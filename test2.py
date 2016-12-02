from __future__ import division  # avoid integer division for python2
from __future__ import print_function  # make python 2.7 and 3 print compatible
from scipy import stats
import matplotlib.pyplot as plt
import numpy


class Pulse:
    """Build Pulse Signal:
        PULSE starts at HIGH value, goes to LOW at start with duration of NEG
        stays LOW until END and then returns to HIGH in POS time counts """

    def __init__(self, time, high=10., low=5., start=60., end=120., pos=5., neg=6.):
        signal = numpy.array([])

        for t in numpy.nditer(time):
            value = 0
            if t < start:
                value = high
            elif t < start + neg:
                # linear slope down
                m = (low - high) / neg
                b = low - m * neg
                value = m * (t - start) + b
            elif t < end - pos:
                value = low
            elif t < end:
                # linear slope up
                m = (low - high) / pos
                b = low - m * pos
                value = m * (end - t) + b
            elif t >= end:
                value = high
            signal = numpy.append(signal, value)
        self.pulse = signal


class Thermal:
    """Generate thermal signal profile, quicker rise than cool"""

    def __init__(self, time, start=45., center=100., offset=3.77):
        signal = numpy.array([])
        # First fit was done with exponential estimation -- not used now
        # Tao = 0.025  # nominal cooling curve
        # Tao1 = Tao * (center / (center - start))  # steeper change for heating
        # peak = 10.53356  # used for exp fit - first attempt

        # second fit was done using plotting paper to estimate curves and
        # choosing the type of fit that produced the least error
        # logarithmic fit for rise and exponential fit for fall

        # would prefer to work from real data, rather than estimate curve types

        for t in numpy.nditer(time):
            if t < center:
                if t < start:
                    value = 0  # RF amp off, no thermal
                else:
                    # WAS: value=peak*math.exp((t-center)*Tao1)
                    # new formula created from power curve fit estimate of Fig.8 where fit is r^2=0.9806578
                    # and a fixed offset of 3.77 (pulse) um was subtracted to remove "offset" contribution
                    # when estimating the curve fit so offset is added back in if needed to create additional
                    # offsets for experimenting
                    value = 17.2198713278 * numpy.log(t) + 1175.584607 - 1249.360 + offset
            else:
                # WAS: value=peak*math.exp((t-center)*Tao*-1)
                # new formula created from power curve fit estimate of Fig.8 where fit is r^2=0.991166
                value = 1341.88491 * numpy.power(t, -0.013793345) - 1249.360
            signal = numpy.append(signal, value)
        self.thermal = signal


class Calc:
    """Compute regression information
        time = time baseline
        x = waveform
        start = time to start analysis
        stop = time to stop analysis
        start2 = if a second part of the waveform should be included, this is the start time
        stop2 = this the stop time for the second segment of the waveform being analyzed

        Generates:
        slope, intercept, r_value, p_value, std_err
        time = arrray containing time data that the linear regresion was computed over
    """

    def __init__(self, time, x, start=0., stop=5., start2=0., stop2=0.):
        self.slope = 0
        self.intercept = 0
        self.r_value = 0
        self.p_value = 0
        self.std_err = 0
        self.time = numpy.array([])
        self.x = numpy.array([])
        index1 = numpy.where((time >= start) & (time <= stop))[0]
        calc1 = x[index1[:]]
        xtimes = time[index1[:]]
        if start2 < stop2:
            # Add second portion of curve if needed for piecewise estimates on calibration pulse tops
            index1b = numpy.where((time >= start2) & (time <= stop2))[0]
            calc1 = numpy.append(calc1, x[index1b[:]])
            xtimes = numpy.append(xtimes, time[index1b[:]])

        self.time = xtimes  # save time array for reference
        self.x = calc1
        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(xtimes, calc1)

    def interp(self):
        """Produce line based on linear fit and time window
            returns array of line and it correspondes to self.time array
        """
        line = numpy.array([])
        for t in numpy.nditer(self.time):
            value = self.slope * t + self.intercept
            line = numpy.append(line, value)
        return line

    def estimate(self, time):
        """Estimate single value of curve fit at a specific time
            returns single point based on computed linear equation parameters
        """
        value = self.slope * time + self.intercept
        return value

    def prnt(self):
        """Print results of linear regression"""
        print('m=', self.slope, 'b=', self.intercept, 'r=', self.r_value, 'p=', self.p_value, 'stderr=', self.std_err)


# TIME array
span = 200  # minutes to run
resolution = 0.5  # in minutes
times = numpy.linspace(0, span, span / resolution + 1, endpoint=True)

# Gaussian noise array
# center around 0 and just guess at standard deviation as it was not
# reported by Eagleworks
mu, sigma = 0.0, 0.03  # mean and standard deviation for noise
noise = numpy.random.normal(mu, sigma, times.size)

# METHOD -- BASED ON Fig. 8
# waveforms are built based on 0 + change in displacement as reported
# in force measurements.  This makes superimposing the waveforms
# easier because they can center on 0 displacement and then be build
# using change in displacement.
#
# The final result can then offset by 1249.360 to get nominal 
# um of displacement 

# From EW paper P. 4, the dx vs df is computed based on their statement:
#   " 0.983 um, which corresponds with the calibration pulse magnitude of 29 uN"
# which means dx/df = 0.0338965517 (this ratio seems to vary in the report?)
#     On P.5 "two fitted linear equations is 1.078 um, which corresponds with the 
#     calibration pulse magnitude of 29 uN."

dx_df = 0.0338965517

cal1_pulse = Pulse(times, high=0, low=-1.078, start=5, end=35, pos=8, neg=5)
cal2_pulse = Pulse(times, high=0, low=-1.078, start=160, end=180, pos=8, neg=5)

# for impluse they compute 106uN or 3.77 um on Equation 1.
# (however using computed dx/df above it's more like 111.2 uN)
# what is right?
impulse = Pulse(times, high=0, low=3.77, start=60, end=101, pos=3, neg=3)

# build composite of 2 pulses
pulse = numpy.add(cal1_pulse.pulse, cal2_pulse.pulse)

# From Fig. 8 using dx=0.13826*t+1245.238 for the thermal pulse slope
# we find at t=105 the peak is 1259.7553 (from careful plotting, 
# on graph paper it appears to be 1259.286 at 107min)
# subtract our offset of 1249.36 then scale is 0->10.3953 for maximum thermal
# then remove the 'impulse' contribution of 3.77
thermal = Thermal(times, start=60, center=107, offset=3.77)

# build composite signal
total = numpy.add(pulse, impulse.pulse)
total = numpy.add(total, noise)
total = numpy.add(total, thermal.thermal)
total = numpy.add(total, 1249.360)  # offset dx to nominal

# REVERSE CALCULATIONS -- Discussion on pp.4-5 gives sample times
# Compute curve filts for Pulse 1 with 2 time Windows
print("Pulse 1 Top")
cal1Top = Calc(times, total, 0, 4.4, 44.6, 57.6)
cal1Top.prnt()

print("Pulse 1 Bottom")
cal1Bot = Calc(times, total, 11.4, 28.6)
cal1Bot.prnt()

# Compute curve filts for Pulse 2 Windows
cal2Top = Calc(times, total, 155, 158.6, 178.8, 184)
print("Pulse 2 Top")
cal2Top.prnt()

print("Pulse 2 Bottom")
cal2Bot = Calc(times, total, 163.2, 171.6)
cal2Bot.prnt()

# Compute Pulse Force
print("Pulse Force")
f_pulse = Calc(times, total, 83.8, 95) # adjusted pulse time window
f_pulse.prnt()


print("CAL1 Pulse Separation: ", )
val = cal1Top.estimate(20.2) - cal1Bot.estimate(20.2)
print(val, " um or ", val / dx_df, " uN force")

print("CAL2 Pulse Separation: ", )
val = cal2Top.estimate(167) - cal2Bot.estimate(167)
print(val, " um or ", val / dx_df, " uN force")

print("Impulse Force Calculations: ", )
val = cal1Top.intercept - f_pulse.intercept 
print(val, " um or ", val / dx_df, " uN force")

# Plot signals
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_subplot(4, 1, 1)
ax.plot(times, impulse.pulse)
ax.set_ylabel('impulse')

ax = fig.add_subplot(4, 1, 2)
ax.plot(times, pulse)
ax.set_ylabel('pulse')

ax = fig.add_subplot(4, 1, 3)
ax.plot(times, thermal.thermal)
ax.set_ylabel('thermal')

ax = fig.add_subplot(4, 1, 4)
ax.plot(times, total)
ax.set_ylabel('total')

# Create larger result plot with linear line estimates added in
fig1 = plt.figure(figsize=(8, 6), dpi=80)
ax = fig1.add_subplot(111)
plt.plot(times, total)
plt.plot(cal1Top.time, cal1Top.interp())
plt.plot(cal1Bot.time, cal1Bot.interp())
plt.plot(f_pulse.time, f_pulse.interp())
plt.plot(cal2Top.time, cal2Top.interp())
plt.plot(cal2Bot.time, cal2Bot.interp())
ax.set_ylabel('total')

fig.savefig('signals_t2.png')
fig1.savefig('combined_t2.png')

# Uncomment if you prefer on-screen plots
#plt.show()

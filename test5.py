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
        self.data = signal


class ForcePulse:
    """Build Force Pulse Signal from Figure 5 model of eagleworks paper:
        PULSE starts at HIGH value, goes to LOW following the timing and slopes given in Figure 5.
        The curves are rescaled in both time and amplitude to fit the desired experimental constraints"""

    def __init__(self, time, high=0, low=3.77, start=60., end=110):
        signal = numpy.array([])

        # rise (0-1):  f(x) = 2.9873202637x + 0.0236544518
        # R^2 = 0.9991409013

        # top (1-4): f(x) = 0.0012770362x + 3.0165657699
        # R^2 = 0.9999989071
        # peak value = 3.0215812262
        peak = 3.0215812262

        # fall (4-5): f(x) =  - 2.9987385162x + 14.9945782273
        # R^2 = 0.998996942

        # elsewhere f(x)=0

        span = end - start  # pulse width
        stepsize = 5 / span
        rise_done = 1 / stepsize
        top_done = 4 / stepsize
        fall_done = 5 / stepsize

        for t in numpy.nditer(time):
            value = 0

            # rescale time
            x = (t - start) * stepsize

            if t < start:
                value = high / peak
            elif t < start + rise_done:
                # linear slope
                value = 2.9873202637 * x + 0.0236544518
                value = value * low / peak
            elif t < start + top_done:
                value = 0.0012770362 * x + 3.0165657699
                value = value * low / peak
            elif t < start + fall_done:
                value = -2.9987385162 * x + 14.9945782273
                value = value * low / peak
            elif t >= start + fall_done:
                value = high / peak

            signal = numpy.append(signal, value)
        self.data = signal


class Thermal:
    """Generate thermal signal profile using Fig 5 curve fit"""

    def __init__(self, time, start=60., center=100, offset=0., peak=1259.7553):
        signal = numpy.array([])
        # full rise-fall curve fit with curve fit from Figure 5.
        # curve fit r^2 = 0.99361

        for t in numpy.nditer(time):
            if t < center:
                if t < start:
                    value = 0  # RF amp off, no thermal
                else:
                    # rise curve
                    # rise curve(0-5):  f(x) =  - 0.0064354826x^4 + 0.0903571004x^3 - 0.5212241274x^2 + 1.947007813x + 0.0530313985
                    # rise x=0 at start, x=5 at center
                    x = (t - start) * 5 / (center - start)
                    value = - 0.0064354826 * numpy.power(x, 4) + 0.0903571004 * numpy.power(x,
                                                                                            3) - 0.5212241274 * numpy.power(
                        x, 2) + 1.947007813 * x + 0.0530313985
            else:
                # fall curve(5-10): f(x) = 0.0065180865x^4 - 0.2227500576x^3 + 2.8971895487x^2 - 17.4937755423x + 42.8137121961
                # fall x=5 at start to x=10 at end (time[-1])
                x = 5 + (t - center) * 5 / (time[-1] - center)
                value = 0.0065180865 * numpy.power(x, 4) - 0.2227500576 * numpy.power(x,
                                                                                      3) + 2.8971895487 * numpy.power(x,
                                                                                                                      2) - 17.4937755423 * x + 42.8137121961
            value = peak / 4 * value  # rescale amplitude
            value = value + offset
            signal = numpy.append(signal, value)
        self.data = signal


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
        time = array containing time data that the linear regression was computed over
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
            returns array of line and it corresponds to self.time array
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
span = 200  # seconds to run
resolution = 0.5  # in seconds
times = numpy.linspace(0, span, span / resolution + 1, endpoint=True)

# Gaussian noise array
# center around 0 and just guess at standard deviation as it was not
# reported by Eagleworks
mu, sigma = 0.0, 0.01  # mean and standard deviation for noise
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

# for impulse force is computed using Fig. 5 model
impulse = ForcePulse(times, high=0, low=3.77, start=58, end=106)

# build composite of 2 pulses
pulse = numpy.add(cal1_pulse.data, cal2_pulse.data)

# From Fig. 8:
# we find at t=106.43 the peak is 1259.3950437986 from digitizing their plot (see ew-graph.csv),
# subtract our baseline offset of 1249.36 then scale is 0->10.035 for maximum thermal
# thermal curve is computed using Figure 5 model
thermal = Thermal(times, start=60, center=107, offset=0, peak=1259.3950437986 - 1249.360)

# build composite signal
total = numpy.add(pulse, impulse.data)
total = numpy.add(total, noise)
total = numpy.add(total, thermal.data)
total = numpy.add(total, 1249.360)  # offset dx to nominal

# REVERSE CALCULATIONS -- Discussion on pp.4-5 gives sample times
# Compute curve filts for Pulse 1 with 2 time Windows
print("Pulse 1 Top")
cal1Top = Calc(times, total, 0, 4.4, 44.6, 57.6)  # EW time window
cal1Top.prnt()

print("Pulse 1 Bottom")
cal1Bot = Calc(times, total, 11.4, 28.6)  # EW time window
cal1Bot.prnt()

# Compute curve filts for Pulse 2 Windows
cal2Top = Calc(times, total, 155, 158.6, 178.8, 184)  # EW time window
print("Pulse 2 Top")
cal2Top.prnt()

print("Pulse 2 Bottom")
cal2Bot = Calc(times, total, 163.2, 171.6)  # EW time window
cal2Bot.prnt()

# Compute Pulse Force
print("Pulse Force")
f_pulse = Calc(times, total, 83.8, 95.)  # EW time window 83.8-100.8
f_pulse.prnt()

print("CAL1 Pulse Separation: ", )
Cal1_dx_val = cal1Top.estimate(20.2) - cal1Bot.estimate(20.2)
print(Cal1_dx_val, " um or ", Cal1_dx_val / dx_df, " uN force")

print("CAL2 Pulse Separation: ", )
Cal2_dx_val = cal2Top.estimate(167) - cal2Bot.estimate(167)
print(Cal2_dx_val, " um or ", Cal2_dx_val / dx_df, " uN force")

print("Impulse Force Calculations: ", )
# compute shifted intercept line using time of 59.0519660294 which
# was reverse calculated from Figure 8. when EW computed 1241.468
# for their shifted offset
shifted_b = (cal1Top.slope - f_pulse.slope) * 59.0519660294 + cal1Top.intercept
val = f_pulse.intercept - shifted_b
print('shifted_b =', shifted_b, ' and Cal1 b=', cal1Top.intercept)
dx_df = ((Cal1_dx_val + Cal2_dx_val) / 2) / 29  # 29uN, but x is normalized from um so E-6 is dropped
print(val, " um or ", val / dx_df, " uN force and dx/df =", dx_df)

# Plot signals
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_subplot(4, 1, 1)
ax.plot(times, impulse.data)
ax.set_ylabel('impulse')

ax = fig.add_subplot(4, 1, 2)
ax.plot(times, pulse)
ax.set_ylabel('pulse')

ax = fig.add_subplot(4, 1, 3)
ax.plot(times, thermal.data)
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

fig.savefig('signals_t5.png')
fig1.savefig('combined_t5.png')

# Uncomment if you prefer on-screen plots
# plt.show()

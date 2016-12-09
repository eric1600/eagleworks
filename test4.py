from __future__ import division  # avoid integer division for python2
from __future__ import print_function  # make python 2.7 and 3 print compatible
from scipy import stats
import matplotlib.pyplot as plt
import numpy
import csv


class LabData:
    """Handle measured data"""

    def __init__(self, time, filename='ew-graph.csv'):
        tdata = numpy.array([])
        xdata = numpy.array([])
        xsample = numpy.array([])

        with open(filename, 'rb') as csvfile:
            filereader = csv.reader(csvfile)
            for row in filereader:
                tdata = numpy.append(tdata, row[0])
                xdata = numpy.append(xdata, row[1])

        tdata = tdata.astype('float64')
        xdata = xdata.astype('float64')

        for t in numpy.nditer(time):
            value = 0
            if t <= tdata[0]:
                xsample = numpy.append(xsample, xdata[0])
            elif t >= tdata[-1]:
                xsample = numpy.append(xsample, xdata[-1])
            else:
                index = numpy.where(tdata > t)
                i = index[0][0]
                xsample = numpy.append(xsample, (numpy.interp(t, [tdata[i - 1], tdata[i]], [xdata[i - 1], xdata[i]])))
        self.labdata = xsample


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

# Load Lab Data
data = LabData(times)

# Based on 29uN as per Figure 8 description
dx_df = 0.0338965517

# Using force pulse model outlined in Figure 5
# low = -2.076 removes all force from thermal+force curve
# low = 0 shows 103uN (vs. expected of 106uN)
# low = -3.77 (or 106uN) yields force of -84.5 uN
# low = -3.7625 (about 106uN using computed dx/df = 0.0354953532734 from calibration pulses on data
impulse = ForcePulse(times, high=0, low=3.7625, start=58, end=115)  # timings for this is approximate

# Take total data from EW data - no composite signals used
total = data.labdata
total = numpy.add(total, impulse.data)

# REVERSE CALCULATIONS -- Discussion on pp.4-5 gives sample times
# Compute curve fits for Pulse 1 with 2 time Windows
print("Pulse 1 Top")
cal1Top = Calc(times, total, 0, 4.4, 44.6, 57.6)  # EW time window
cal1Top.prnt()
print('Eagleworks:  m= 0.004615 b=1249.360 errors m_err=', 0.004615 - cal1Top.slope, 'b_err=',
      1249.360 - cal1Top.intercept)

print("Pulse 1 Bottom")
cal1Bot = Calc(times, total, 11.4, 28.6)  # EW time window
cal1Bot.prnt()
print('Eagleworks:  m= 0.005096 b=1248.367 errors m_err=', 0.005096 - cal1Bot.slope, 'b_err=',
      1248.367 - cal1Bot.intercept)

# Compute curve fits for Pulse 2 Windows
cal2Top = Calc(times, total, 155, 158.6, 178.8, 184)  # EW time window
print("Pulse 2 Top")
cal2Top.prnt()
print('Eagleworks:  m= -0.07825 b=1263.499  errors m_err=', -0.07825 - cal2Top.slope, 'b_err=',
      1263.499 - cal2Top.intercept)

print("Pulse 2 Bottom")
cal2Bot = Calc(times, total, 163.2, 171.6)  # EW time window
cal2Bot.prnt()
print('Eagleworks:  m= -0.0827 b=1263.163   errors m_err=', -0.0827 - cal2Bot.slope, 'b_err=',
      1263.163 - cal2Bot.intercept)

# Compute Pulse Force
print("Pulse Force")
f_pulse = Calc(times, total, 83.8, 102.8)  # EW time window
f_pulse.prnt()
print('Eagleworks:  m=0.13826 b=1245.238  errors m_err=', 0.13826 - f_pulse.slope, 'b_err=',
      1245.238 - f_pulse.intercept)

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
print('Eagleworks shifted_b=1241.468 error =', 1241.468 - shifted_b)
dx_df = ((Cal1_dx_val + Cal2_dx_val) / 2) / 29  # 29uN, but x is normalized from um so E-6 is dropped
print(val, " um or ", val / dx_df, " uN force and dx/df =", dx_df)

# Plot signals
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_subplot(2, 1, 1)
ax.plot(times, impulse.data)
ax.set_ylabel('impulse')

ax = fig.add_subplot(2, 1, 2)
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

fig.savefig('signals_t4.png')
fig1.savefig('combined_t4.png')

# Uncomment if you prefer on-screen plots
# plt.show()

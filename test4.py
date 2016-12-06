from __future__ import division  # avoid integer division for python2
from __future__ import print_function  # make python 2.7 and 3 print compatible
from scipy import stats
import matplotlib.pyplot as plt
import numpy
import csv

class LabData:
	"""Handle measured data"""
	
	def __init__(self,time,filename='ew-graph.csv'):
		tdata=numpy.array([])
		xdata=numpy.array([])
		xsample=numpy.array([])

		with open(filename, 'rb') as csvfile:
	    		filereader = csv.reader(csvfile)
			for row in filereader:
				tdata=numpy.append(tdata,row[0])
				xdata=numpy.append(xdata,row[1])

		tdata=tdata.astype('float64')
		xdata=xdata.astype('float64')
		
		for t in numpy.nditer(time):
			value=0
			if t <= tdata[0]:
				xsample=numpy.append(xsample,xdata[0])
			elif t>=tdata[-1]:
				xsample=numpy.append(xsample,xdata[-1])
			else:
				index=numpy.where(tdata>t)
				i=index[0][0]
				xsample=numpy.append(xsample,(numpy.interp(t, [tdata[i-1], tdata[i]],[xdata[i-1],xdata[i]])))
		self.labdata=xsample
				
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

	# For reference, but not used:
	# Eagleworks thermal model both curves fit r^2>0.999 (From Fig. 5 in their paper)
	# rise curve(0-5):  f(x) =  - 0.0064354826x^4 + 0.0903571004x^3 - 0.5212241274x^2 + 1.947007813x + 0.0530313985
	# fall curve(5-10): f(x) = 0.0065180865x^4 - 0.2227500576x^3 + 2.8971895487x^2 - 17.4937755423x + 42.8137121961

    def __init__(self, time, start=45., offset=-1249.360):
        signal = numpy.array([])
        # full rise-fall curve fit with following error estimates:
        # r^2 = 0.992116
        #       um                          uN
        # max   0.6901810304                20.3613935709
        # min  -0.4266644384               -12.5872519989
        # ave   1.48698258928764E-05         0.0004386826
        # stdev 0.3454826627                10.1922657362

        # curve fit r^2 = 0.99361

        for t in numpy.nditer(time):
            if t < start:
                value = 0  # RF amp off, no thermal
            else:
                # sixth order poly fit
                value = 0.000000000071282927264184 * numpy.power(t, 6)
                value = value - 0.0000000616854969614949 * numpy.power(t, 5)
                value = value + 0.0000214062587544044 * numpy.power(t, 4)
                value = value - 0.0037728812 * numpy.power(t, 3)
                value = value + 0.3506329937 * numpy.power(t, 2)
                value = value + -15.9748263936 * t
                value = value + 1527.7174983854 + offset
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
span = 200  # seconds to run
resolution = 0.5  # in seconds
times = numpy.linspace(0, span, span / resolution + 1, endpoint=True)

# Load Lab Data
# resample to times array
data=LabData(times)

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
impulse = Pulse(times, high=0, low=0, start=60, end=101, pos=3, neg=3)

# build composite of 2 pulses
pulse = numpy.add(cal1_pulse.pulse, cal2_pulse.pulse)

# From Fig. 8 using dx=0.13826*t+1245.238 for the thermal pulse slope
# we find at t=105 the peak is 1259.7553 (from careful plotting, 
# on graph paper it appears to be 1259.286 at 107min)
# subtract our offset of 1249.36 then scale is 0->10.3953 for maximum thermal
# then remove the 'impulse' contribution of 3.77
thermal = Thermal(times, start=60, offset=-1249.360)

# build composite signal
#total = numpy.add(pulse, impulse.pulse)
#total = numpy.add(total, noise)
#total = numpy.add(total, thermal.thermal)
#total = numpy.add(total, 1249.360)  # offset dx to nominal

# Take total data from EW data - no composite signals used
total=data.labdata

# REVERSE CALCULATIONS -- Discussion on pp.4-5 gives sample times
# Compute curve filts for Pulse 1 with 2 time Windows
print("Pulse 1 Top")
cal1Top = Calc(times, total, 0, 4.4, 44.6, 57.6)  # EW time window
cal1Top.prnt()
print('Eagleworks:  m= 0.004615 b=1249.360 errors m_err=', 0.004615-cal1Top.slope, 'b_err=',1249.360-cal1Top.intercept)

print("Pulse 1 Bottom")
cal1Bot = Calc(times, total, 11.4, 28.6)  # EW time window
cal1Bot.prnt()
print('Eagleworks:  m= 0.005096 b=1248.367 errors m_err=', 0.005096-cal1Bot.slope, 'b_err=',1248.367-cal1Bot.intercept)

# Compute curve filts for Pulse 2 Windows
cal2Top = Calc(times, total, 155, 158.6, 178.8, 184)  # EW time window
print("Pulse 2 Top")
cal2Top.prnt()
print('Eagleworks:  m= -0.07825 b=1263.499  errors m_err=', -0.07825-cal2Top.slope, 'b_err=',1263.499-cal2Top.intercept)

print("Pulse 2 Bottom")
cal2Bot = Calc(times, total, 163.2, 171.6)  # EW time window
cal2Bot.prnt()
print('Eagleworks:  m= -0.0827 b=1263.163   errors m_err=', -0.0827-cal2Bot.slope, 'b_err=',1263.163 -cal2Bot.intercept)

# Compute Pulse Force
print("Pulse Force")
f_pulse = Calc(times, total, 83.8, 102.8)  # EW time window
f_pulse.prnt()
print('Eagleworks:  m=0.13826 b=1245.238  errors m_err=',0.13826-f_pulse.slope, 'b_err=',1245.238 -f_pulse.intercept)

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
print ('shifted_b =',shifted_b, ' and Cal1 b=',cal1Top.intercept)
print('Eagleworks shifted_b=1241.468 error =',1241.468-shifted_b)
dx_df = ((Cal1_dx_val + Cal2_dx_val) / 2) / 29 # 29uN, but x is normalized from um so E-6 is dropped
print(val, " um or ", val / dx_df, " uN force and dx/df =", dx_df)

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

fig.savefig('signals_t4.png')
fig1.savefig('combined_t4.png')

# Uncomment if you prefer on-screen plots
#plt.show()

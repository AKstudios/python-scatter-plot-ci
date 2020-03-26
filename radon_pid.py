# -*- coding: utf-8 -*-
# ===========================================================================
# This code plots outdoor temperature vs radiator on % for various Control
# strategies.

# Developed by Akram Ali
# Updated on: 03/26/2020

# ===========================================================================
# import libraries
# ===========================================================================

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy
from scipy import stats
import statsmodels.api as sm
from scipy import stats
# import seaborn as sns

# ===========================================================================
# define data lists and ranges
# ===========================================================================

radon = [20.2, 47.6, 68.3, 68.9, 49.8, 57.5, 73.4, 46.4, 46.1, 51.0, 48.2, 
49.3, 39.7, 34.7, 27.0, 54.8, 63.3, 56.9, 48.9, 36.6, 28.3, 58.5, 38.3, 49.3, 
50.4, 49.3, 27.3, 19.4, 17.0, 4.7, 6.4, 11.4, 21.3, 18.8, 15.8, 5.6, 19.7, 
19.7, 14.6, 13.2, 13.2, 13.9]

tout = [3.82, 1.52, -1.43, -1.64, -1.04, -0.80, -0.34, 1.84, 1.62, -4.64,
-7.54, -3.77, 3.29, 6.38, 5.70, 4.37, 5.27, 5.82, 1.06, 3.57, 11.52, -0.80,
-0.99, -0.75, -7.75, -14.44, -4.44, -0.97, 5.70, 4.37, 5.27, 5.82, 1.06, 3.57,
11.52, 11.98, 4.01, 4.54, 8.31, 4.69, 2.58, 2.13]

a = [0,21]   # a and b indicate the start and end index of each strategy
b = [20,41]

# ===========================================================================
# configure graph display
# ===========================================================================

# set font family
hfont = {'family':'Arial'}
plt.rcParams.update({'font.family': 'Arial', 'font.size':12})

# set axis minor tick marks
plt.axes().yaxis.set_minor_locator(mtick.MultipleLocator(4))
plt.axes().xaxis.set_minor_locator(mtick.MultipleLocator(2))
plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

# config axis labels
plt.xlabel("Outdoor Temperature (\u00b0C)", **hfont)
plt.ylabel("Radiator usage (%)", **hfont)

# set line and marker types and colors
marker = ['o', 'D']
marker_facecolor = ['#ffbf00', '#ffffff']
names = ['PID, R\u00b2= ', 'PID + Occupancy, R\u00b2= ']
lines = ['-.','--']
line_spacing = [[1, 5], [5, 5]]
transparency = [1, 0.7]

# ===========================================================================
# iterate over both lists
# ===========================================================================

for n, (i,j) in enumerate(zip(a,b)):
    # get x and y
    x = numpy.array(tout[i:j])
    y = numpy.array(radon[i:j])

    # set axis ranges
    plt.ylim(0, 100)
    plt.xlim(-30, 20)

    # perform regressions
    z = numpy.polyfit(x, y, 1)
    p = numpy.poly1d(z)

    # Generated linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    r2 = round(r_value**2, 4)
    line = slope * x + intercept

    # plot data
    plt.plot(x, y, marker[n],markeredgewidth=0.75
    ,markeredgecolor='k',markerfacecolor=marker_facecolor[n],
    alpha=transparency[n],label='%s%s' % (names[n], str(r2)))

    # plot trendline
    order = numpy.argsort(x)
    plt.plot(x[order],p(x[order]),'k%s' % lines[n], linewidth=0.85)
    # plt.plot(x,line,'k%s' % lines[n], linewidth=0.85)

    # do some magic statistics
    # sns.regplot(x,y)
    x = sm.add_constant(x)  # Adds a column of 1s to the array
    model = sm.OLS(y, x)    # A simple Ordinary Least Squares model
    fitted = model.fit()    # Full fit of the model
    y_hat = fitted.predict(x)   # Return linear predicted values from a design
                                # matrix. x is the array from add_constant line
    x_pred = numpy.linspace(x.min(), x.max(), 50) # Return evenly spaced numbers
                                                # over the specified interval.
    x_pred2 = sm.add_constant(x_pred)
    y_pred = fitted.predict(x_pred2)
    y_err = y - y_hat
    mean_x = x.T[1].mean()
    l = len(x)
    dof = l - fitted.df_model - 1   # degrees of freedom
    t = stats.t.ppf(1-0.025, df=dof) # A Student’s T continuous random variable
    s_err = numpy.sum(numpy.power(y_err, 2))
    conf = t * numpy.sqrt((s_err/(l-2))*(1.0/l + (numpy.power((x_pred-mean_x),2)
     / ((numpy.sum(numpy.power(x_pred,2))) - l*(numpy.power(mean_x,2))))))
    upper = y_pred + abs(conf)
    lower = y_pred - abs(conf)

    # plot calculated 95% confidence interval bands
    plt.fill_between(x_pred, lower, upper, color='#888888', alpha=0.25)

# ===========================================================================
# configure legend
# ===========================================================================

plt.rc('legend', fontsize=9)
plt.legend(loc='best')      # legend location
leg = plt.legend()
leg.get_frame().set_edgecolor('k')

# ===========================================================================
# save and show plot
# ===========================================================================
plt.savefig('tout_vs_radon_pid.png', dpi=300)
plt.show()

# Project Timeline

**General rule of thumb:** Programs do get better as the date changes from mid-June to mid-July.

## 11 Jun
**program.py:**
- Calculating alm values and storing in text files and simply plotting a predownloaded fits file.

**test.py:**
- Defining simple functions of b and not knowing how to integrate it with the alm calculation part.

**testanalytical.py:**
- Same as above, terrible attempts at calculating alm values for a given function.

**testanalyticalreal.py:**
- Finally managed to somewhat integrate the function B into the calculation of alms.

## 13 Jun
**contourf-sintheta.py:**
- First reconstruction and plotting using the calculated ylm, alm values.

**trial-rename.py:**
- Takes fits file, calcs alm, stores in text file, plots original cr map.

## 14 Jun
**contourf-sintheta2.py:**
- Plots original function, calculates alm for given lmax, can reconstruct and replot as well.

**contourf-sintheta2xy.py:**
- Usage of subplots to plot both og function as well as recons function.

**contourf-sintheta2xymultiplots.py:**
- Bad alternate to the above? Ig?

**contourf-siny.py:**
- Calcs alm, reconstructs, plots both one after the other.

**contourfmultiplots.py:**
- Bad experimenting with the plotting style before I arrived at subplots method.

**contourf-subplots.py:**
- Subplotting, but more primitive to the contourf-sintheta2xy.py.

## 18 Jun
**pw.py:**
- Neat code that calcs alms up to given lmax, recons, and plots both in subplots.

**pw2.py:**
- Same as above with additional functionality of being able to automatically iterate through a set of values of lmax and num_points.

## 19 Jun
**pw.py:**
- Same as pw2.py of 18 Jun.

## 20 Jun
**pw.py:**
- Same as pw.py of 19 Jun.

**pw2.py:**
- Same as above.

**pw3.py:**
- Same as above with added functionality of difference/delta being plotted as well in new subplot.

## 21 Jun
**pw.py:**
- Same as above with added functionalities of a stopwatch and different variations of the plotty function.

**pw2.py:**
- Same as above with added functionality of reading from stored csv files in case of re-running with the same parametric values instead of recalculating the same set of alm values.

**pw3.py:**
- Same as above with added functionalities of a very bad error metric and confirmation of retrieval of data from the pre-existing alm values csv file.

**pw3helper.py:**
- Plots the variation of total(abs(delta)) vs the num_points parameter for a given lmax, with a stopwatch.

**pw3helper2.py:**
- Seems to be in fact a smaller version of the above code.

**pw3helpershelper.py:**
- Just precomputes all the alm values for the given set of num_points parameters and the lmax parameter and stores in the csv file.

**pw3helpershelper2.py:**
- Automates the range in which the num_points parameters vary, instead of asking for user input; keeping lmax constant.

## 22 Jun
**pw-siny.py:**
- First usage of dblquad for the integration process, other than that seems to be the same as pw3.py.

**pw.py:**
- Seems to be the same as above.

**pw2.py:**
- Same as above with a better way of storing alm values and then reading from them in case of re-running of the code with the same values.

**pw2-siny.py:**
- Same as above but there are beeps to show completion of different parts of the code using winsound and there is also a display of all the parameters before the code starts running, to ask for confirmation from the user.

## 23 Jun
**pw.py:**
- Seems to be the same as above with a slightly better UI.

**pw2.py:**
- Same as above with a way to change the function being used by changing the string called func_name.

**pwlam.py:**
- Trying to make the code to be able to run for different functions - B by using lambda functions here.

**pwload.py:**
- Same as pw2.py but now there's a simple progress bar to show how much of the alm calculation is done.

**pwload.py:**
- Same as above but with a slightly better progress bar.

## 24 Jun
**pw.py:**
- Same as the best of 23 Jun, but with added error_folder to store a text file with the total absolute error and a csv file to store all alm values.

**pwerr.py:**
- Computes the total absolute error for a range of num_points values and the same lmax.

**pwf(x).py:**
- Trying to make the code work for functions of x as well.

**pwf(y).py:**
- I think it's the same as above, but instead of x it's y.

**pwsinx.py:**
- Same as pw.py of 24 Jun but it's sin(x) instead of sin(y).

**pwvaryyx.py:**
- Trying to make the code work no matter what I return from the b(y, x) function.

## 25 Jun
**pw.py:**
- Still trying to make it work for different functions.

**pwf(x).py:**
- Still trying to make it work for different functions.

**Literally every other python file in this folder has the same job description. I'm skipping. Do I turn out to be successful in those quests? Try running the code to figure out; even I forgot lol.**

## 26 Jun
**26.py:**
- First attempt on this day. Reads from b.txt and considers that np expression that matches with the string in the code.

**pw.py:**
- Runs for particular function.

**pwcomb.py:**
- Runs for all the functions in b.txt and compares max run times and p errors for all.

**pwfil.py:**
- Trying to apply Gaussian filter to the input data.

**pwfits-29Jun.py:**
- Plotting a fits data instead of the function data.

**pwfits.py:**
- Same as above.

**pwfitsgauss-varcrmap.py:**
- Can change the cr map number and it finds its fits file and does the deed. By the way, I switched to the normal integration instead of the adaptive thingy because it isn't as good at estimating the intervals.

**pwfitsgauss-varcrmapcombo-simpsons.py:**
- Same as above along with red-blue coloring that I figured was better and also I introduced a different type of error metrics somewhere in these codes that belong to this date. Changed the method of integration to Simpson's 1/3.

**pwfitsgauss-varcrmapcombo.py:**
- Same as above but normal integration instead of Simpson's 1/3.

**pwfitsgauss.py:**
- Applying Gaussian filter to the fits input data instead of the b function data.

**pwfitsresolve.py:**
- Tried to use reduced resolution instead of applying Gaussian filter but the filter works better.

**pwsimp.py:**
- Trying out Simpson's 1/3 rule for integration.

## 03 Jul
## 10 Jul
## 11 Jul
## 12 Jul
## 13 Jul
## 16 Jul

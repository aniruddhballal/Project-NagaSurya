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


# Project Timeline

## 03 Jul
**fluxcalc.py:**
- Computes the b_avg, hemispherical_avg, and the polar_avg flux and stores it in flux CSV files per CR map for reuse later.

**pwflux.py:**
- Computes total magnetic flux for every Carrington map and plots the variation of it over the years.

**gecko.py:**
- Makes use of Selenium and Firefox browser to download all the fits files from the HMI Stanford website (dimensions - 720*360).

**pw.py:**
- Best of the lot as of yet. Makes use of Simpson's 1/3 and reconstructs from existing alm files or does the alm calculation along with it and calculates the improvised error metrics and saves the plot. Makes use of a mapping from CR map numbers to the month-year combination. Moves the done fits files to the done folder. Has the normal progress bar and the usual beeps.

**pwbetter.py:**
- Same as above but implemented the rich progress bar instead of tqdm; and it makes use of the `sunpy.coordinates.sun - carrington_rotation_time` to get the month-year from the CR map number now.

**pwbetterer.py:**
- Exactly the same code as above but used for alm calculations only and commented the other parts out.

**pwmotion.py:**
- Basically the same as the above code except it plots only the reconstructed map and does it for lmax = a low value as set by the user to be able to see the large scale features in the map to get an idea of the sense of motion of those same large scale structures.

**main_script.py:**
- Broke the pwbetter.py into this and the supervisor.py code such that this code runs for one CR map number and then the supervisor.py aborts the execution of this program and then calls it again, on an endless loop until it reaches the end of the list of the crmaps in the crmaps.txt file.

**supervisor.py:**
- Supervises the main_script.py or the shortmainscript.py as per user's needs.

**shortmainscript.py:**
- A shortened version of the main_script.py such that only the alm calculation and saving part is there.

## 10 Jul
**avgalm_vs_time.py:**
- Plots the average value of the alm magnitudes in a given percentile of the total alm magnitudes, against the years. There is an option of enabling the Gaussian smoothing on the datapoints being plotted.

**comboplot.py:**
- An attempt at trying to plot the alms against l-m axes along with a viridis colourbar to represent their magnitudes, highlighting the max, second max and the avg magnitudes. One for each CR map.

## 11 Jul
**combo.py:**
- Plots original map, the alm magnitude distribution, the map saturated to +/-200, and then the map saturated to +/-700 in 4 subplots.

**combo2.py:**
- Plots the atomic orbital like spherical harmonic visualization of the max magnitude alm along with the distribution of the alm magnitudes in 2 subplots.

**combo12.py:**
- Plots the original map, the same map saturated to +/-200 and then the atomic orbital like spherical harmonic visualization of the max magnitude alm along with the alm magnitude distribution in 4 subplots. Pretty primitive.

**combo12-2.py:**
- Same thing as above, probably a better visual plot.

**combo12better.py:**
- Plots the alm magnitude distribution, and the reconstructed map with lmax = l of the max magnitude alm, the atomic orbital like visual representation of that particular spherical harmonic with the max magnitude of alm in that distribution, and the original map saturated to +/-200 in 4 subplots.

**combo12better-2.py:**
- Somewhat a better plot than the previous one, the idea stays the same though.

**varylmaxrecons_almvaldist_sat200_top2ylms.py:**
- The name of the code says it all. It's the same as the previous code (combo12better) but now instead of considering the max alm magnitude's mode, it also considers the second max and whichever has a higher value of l, it reconstructs up to that value of l. Plots two atomic orbital like visual representation of the two spherical harmonics.

**sph_harm.py:**
- Simple code that plots the visual representation of a particular ylm.

**lmax_vs_time.py:**
- Plots the l (corresponding to the max magnitude alm) as a function of maps/month-years.

**dipole.py:**
- Very similar to the pwmotion.py from earlier except here the lmax value is fixed at 1 to get an idea of the large scale dipole structure present on the surface of the sun.

**dipole_alm_mag.py:**
- Basically the same as comboplot.py from 10 Jul except the alm values it's reading is fixed to l = 1. So only three modes (1,-1), (1,0), and (1,1) will get plotted in the alm distribution map. Or something like that.

**summation_alm_vs_l.py:**
- Plots the sum of all the alm magnitudes for that particular value of l. One for each CR map. I divide the sum by (2l+1) as an attempt to compute the average alm value for every l.

## 12 Jul
**12345.py:**
- Plots the variation of the magnitudes of modes (1,0), (2,0), (1,0), (2,0) and (5,0) over the years.

**dipole.py:**
- Same as pwbetter.py from 03 Jul except the lmax is fixed at 1 instead of 85.

**pw-weakercounterpart.py:**
- Mostly the same as varylmaxrecons_almvaldist_sat200_top2ylms.py from 11 Jul.

**pw.py:**
- Same as the above python code but now it plots the spherical representation of the two spherical harmonic functions along with the atomic orbital like representation.

**pwbetter.py:**
- Same as above but with better and improved visualizations in the plot.

**sph.py:**
- Plots the visual spherical representation of that particular spherical harmonic, unlike the older atomic orbital like representation.

## 13 Jul
**pw.py:**
- Same as pwbetter.py from 11 Jul but now it also shows what fraction of the average flux is the reconstructed map consisting of. Basically the FINAL VERSION. Following codes are all side quests sort of a thing.

**com.py:**
- My first attempt at "centre of mass". Plots the variation of Gaussian smoothed (lmean, mmean) - one datapoint per map - over the years. Prints the sum of the alm along with the mean (l,m).

**comm.py:**
- Plots the COM in the alm distribution plot - one per CR map and saves all these plots. Does something regarding the closest index and stuff. Doesn't actually get the right COM alm magnitude, does it?

**cheatcom.py:**
- Idea was that this would be the same as COM but now considering datapoints above a given threshold instead of all the datapoints. But is this code actually about that? Yes, it does plot those points in the alm magnitude distribution map - one for each CR map. Threshold can be modified by user.

**cheatcomvstime.py:**
- Idea was to see the evolution of the value that the above code gives, over time. Yes it does that. There is an option for Gaussian smoothing of the datapoints before plotting them.

**l_max_vs_time.py:**
- Plots the variation of the value of l corresponding to the max magnitude of alm for that CR map, over all the maps/years.

**summation_alm_vs_l.py:**
- An attempt at bettering the summation_alm_vs_l.py, by taking into consideration only the last 5 alm values for every value of l - produces one plot for each CR map. Makes sense, because as l values go higher and higher, only the last few modes would have higher magnitudes so should average those values out instead of averaging over the whole set of modes for that l.

## 16 Jul
**cheatcom_time.py:**
- Plots the l corresponding to the cheat COM (above threshold means cheat) along with the average flux over the years. Can apply Gaussian smoothing or can plot without it too.

**com_time_b_avg.py:**
- Plots the l corresponding to the COM along with the average flux over the years. Optional Gaussian smoothing available.

**magcheatcom_time.py:**
- Plots the magnitude of the cheat COM along with the average flux over the years - later realized that I'm plotting the average of all the alms for that CR map ahahahaha.

**magcom_time_b_avg.py:**
- Plots the magnitude of the COM along with the average flux over the years - later realized that I'm plotting the average of all the alms for that CR map ahahahaha.

**com_time_b_tot.py:**
- Same as com_time_b_avg.py except here I'm making use of total magnetic flux instead of average magnetic flux.

**magcom_time_b_tot.py:**
- Same as magcom_time_b_avg.py except here I'm making use of total magnetic flux instead of average magnetic flux.

**fractionalmagcomtimebtot.py:**
- Same as above (magcom_time_b_tot.py) as the name suggests, except now it plots in a different directory the fractional change of both the parameters over time, to be able to compare what fraction of change in b_tot caused what fraction of change in the COM's magnitude. Last point to note is that the magnitude of the COM is basically the average of all the alm values for that l.

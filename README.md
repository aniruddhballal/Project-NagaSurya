General rule of thumb: Programs do get better as the date changes from mid-June to mid-July.

11Jun:
    program.py:
        calculating alm values and storing in text files and simply plotting a predownloaded fits file
    test.py:
        defining simple functions of b and not knowing how to integrate it with the alm calculation part
    testanalytical.py:
        same as above, terrible attempts at calculating alm values for a given function
    testanalyticalreal.py:
        finally managed to somewhat integrate the function B into calculation of alms

13Jun:
    contourf-sintheta.py:
        first reconstruction and plotting using the calculated ylm, alm values
    trial-rename.py:
        takes fits file, calcs alm, stores in text file, plots original cr map

14Jun:
    contourf-sintheta2.py:
        plots original function, calculates alm for given lmax, can reconstruct and replot aswell
    contourf-sintheta2xy.py:
        usage of subplots to plot both og function as well as recons function
    contourf-sintheta2xymultiplots.py:
        bad alternate to the above? ig?
    contourf-siny.py:
        calcs alm, reconstructs, plots both one after the other
    contourfmultiplots.py:
        bad experimenting with the plotting style before i arrived at subplots method
    contourf-subplots.py:
        subplotting, but more primitive to the contourf-sintheta2xy.py

18Jun:
    pw.py:
        neat code that calcs alms upto given lmax, recons, and plots both in subplots
    pw2.py:
        same as above with additional functionality of being able to automatically iterate through a set of values of lmax and num_points

19Jun:
    pw.py:
        same as pw2.py of 18Jun

20Jun:
    pw.py:
        same as pw.py of 19Jun
    pw2.py:
        same as above
    pw3.py:
        same as above with added functionality of difference/delta being plotted aswell in new subplot

21Jun:
    pw.py:
        same as above with added functionalities of a stopwatch and different variations of the plotty function
    pw2.py:
        same as above with added functionality of reading from stored csv files in case of re-running with the same parametric values instead of recalculating the same set of alm values
    pw3.py:
        same as above with added functionalities of a very bad error metric and confirmation of retreival of data from the pre-existing alm values csv file
    pw3helper.py:
        plots the variation of total(abs(delta)) vs the num_points parameter for a given lmax, with a stopwatch
    pw3helper2.py:
        seems to be in fact a smaller version of the above code
    pw3helpershelper.py:
        just precomputes all the alm values for the given set of num_points parameters and the lmax parameter and stores in the csv file
    pw3helpershelper2.py:
        automates the range in which the num_points parameters vary, instead of askign for user input; keeping lmax constant
22Jun:
    pw-siny.py:
        first usage of dblquad for the integration process, other than that seems to be the same as pw3.py
    pw.py:
        seems to be the same as above
    pw2.py:
        same as above with a better way of storing alm values and then reading from them in case of re-running of the code with the same values
    pw2-siny.py:
        same as above but there are beeps to show completion of different parts of the code using winsound and there is also a display of all the parameters before the code starts running, to ask for confirmation from the user
23Jun:
    pw.py:
        seems to be the same as above with a slightly better UI
    pw2.py:
        same as above with a way to change the function being used by changing the string called func_name
    pwlam.py:
        trying to make the code to be able to run for different functions - B by usign lambda functions here
    pwload.py:
        same as pw2.py but now theres a simple progress bar to show how much of the alm calculation is done
    pwload.py:
        same as above but with a slightly better progress bar
24Jun:
    pw.py:
        same as the best of 23Jun, but with added error_folder to store a textfile with the total absolute error and a csvs file to store all alm values
    pwerr.py:
        computes the total absolute error for a range of num_points values and the same lmax
    pwf(x).py:
        trying to make the code work for functions of x aswell
    pwf(y).py:
        i think its the same as above, but instead of x its y
    pwsinx.py:
        same as pw.py of 24Jun but its sin(x) instead of sin(y)
    pwvaryyx.py:
        trying to make the code work no matter what i return from the b(y, x) function
25Jun:
    pw.py:
        still trying to make it work for different functions
    pwf(x).py:
        still trying to make it work for different functions
    literally every other python file in this folder has the same job description. im skipping.
    do i turn out to be successful in those quests? try running the code to figure out even i forgot lol.
26Jun:
    26.py:
        first attempt on this day ig. reads from b.txt and considers that np expression that matches with the string in the code
    pw.py:
        runs for particular function
    pwcomb.py:
        runs for all the functions in b.txt and compares maxruntimes and perrors for all
    pwfil.py:
        trying to apply gaussian filter to the input data
    pwfits-29Jun.py:
        plotting a fits data instead of the function data
    pwfits.py:
        same as above ig
    pwfitsgauss-varcrmap.py:
        can change the cr map number and it finds its fits file and does the deed. by the way i switched to the normal integration instead of the adaptive thingy because it isnt as good at estimating the intervals
    pwfitsgauss-varcrmapcombo-simpsons.py:
        same as above along with red blue colouring that i figured was better and also i introduced a different type of error metrics somewhere in these codes that belong to this date. changed method of integration to simpsons 1/3
    pwfitsgauss-varcrmapcombo.py:
        same as above but normal integration instead of simpsons 1/3
    pwfitsgauss.py:
        applying gaussian filter to the fits input data instead of the b function data
    pwfitsresolve.py:
        tried to use reduced resolution instead of applying gaussian filter but the filter works better
    pwsimp.py:
        trying out simpsons 1/3 rule for integration
03Jul:
10Jul:
11Jul:
12Jul:
13Jul:
16Jul:

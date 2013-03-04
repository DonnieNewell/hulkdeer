
# Course: APMA 3501/STAT 3080/APMA 6548
# Assignment number: 2

# Work in the assigned teams of 2 (same teams as for Homework 1)

# Authors: Donald Newell, Muhammad Yanhaona
# Date: 14Sep2012

# Group Pledge: On our honor as students, we have neither given nor received aid on this assignment.

# Save your R file using the following naming convention (please note the .R extension):
# Lastname1_Lastname2_HW#.R 

# Save all graphs, if any, in a single PDF file using this naming convention:
# Lastname1_Lastname2_Graphs_HW#.pdf 
# Start by writing your names in the graphs file you create.

# Submit your R file and PDF file via Collab (one submission per team; either team member can submit)

# Use this format for all assignments
# Separate problems with a string of many #### as shown below

# List problem number as a comment (see example below)

# Type out each question within the problem as a comment

# Show R command (Not as a comment - because the TA will run your program)

# After each command for which there is an output in the R console
# first have a line saying # Answer. In the next line,
# cut-and-paste answer from R Console into the program below
# and place a # symbol in front of the answer
# as shown below so that this is not
# executed when the whole code for the problem is selected for the 
# Run operation.

# Insert comment symbol and provide answers to discussion
# questions 

###########################################
# Problem 1: Textbook Problem 2.8
library(UsingR)
attach(npdb)
sort(table(state))
detach(npdb)
#ANSWER: state
#AS   GU   DC   DE   AK   VT   ME   NH   SD   RI   HI   ID   WY   ND   MT   AL   CT   WI   PR   NE   NV   IN   NM   WV 
#1    1   11   11   13   14   17   17   17   18   23   23   26   30   37   39   41   43   44   46   47   48   50   55 
# AR   UT   SC   MA   KS   MN   MS   OR   IA   MD   LA   OK   NC   NJ   VA   MO   TN   CO   IL   GA   AZ   WA   KY   MI 
# 56   64   65   68   75   76   79   80   87   90   99  100  106  108  110  120  120  123  135  148  153  160  171  179 
# PA   OH   NY   TX   FL   CA 
# 196  252  353  442  744 1566 
##########################################

###########################################
# Problem 2: Textbook Problem 2.16
total_rivers = length(rivers)
under_five_hundred = length(which(rivers < 500))
proportion = under_five_hundred / total_rivers
proportion

# ANSWER:0.5815603 of the rivers are under 500 miles long

avg = mean(rivers)
under_mean = length(which(rivers < avg))
proportion = under_mean / total_rivers
proportion
# ANSWER: 0.6666667 of the rivers under the mean length

quantile = .75
position = 1 + quantile * (length(rivers) - 1)
sorted_rivers = sort(rivers)
sorted_rivers[position]
# ANSWER: 680
##########################################

###########################################
# Problem 3: Textbook Problem 2.18

avg = mean(rivers)
med = median(rivers)
trimmed_mean = mean(rivers, trim = .25)
hist(rivers)
# ANSWER: mean:591.18 median:425 trimmed_mean:449.915
# There is a relatively large difference between the mean and both the trimmed mean and median. 
# This makes sense since the histogram is skewed right, trimming the data will reduce the outliers and shift
#    the mean towards the median.
##########################################


###########################################
# Problem 4: Textbook Problem 2.31

x1 = rnorm(100)
x2 = rnorm(100)
hist(x1)
hist(x2)

# ANSWER: No, we do not get the exact same histogram.
##########################################

###########################################
# Problem 5:

library(e1071)
help(skewness)
statdat<-read.table("StatGrades.txt",header=TRUE)
attach(statdat)
skewness(HW, type=1)
skewness(final, type=1)

#(a) ANSWER: the skewness(HW) = -1.807552, skewness(final) = -1.346309
#             the negative skew values indicate that the HW and final grades
#             are left skewed. This is shown in the histogram plots.

#(b)
# Number 1 - study images of Devore textbook

# Number 2
par(mfcol=c(2,2))
hist(HW)
hist(final)
qqnorm(HW)
qqnorm(final)

# Number 3
par(mfcol=c(2,3))
norm = rnorm(1000)
lognorm = rlnorm(1000)
hist(norm, main="Histogram of std normal dist")
hist(lognorm, main="Histogram of lognormal dist")
qqnorm(norm, main="qqnorm of normal dist")
qqnorm(lognorm, main="qqnorm of lognormal dist")
qqnorm(HW, main="qqnorm of HW")
qqnorm(final, main="qqnorm of final")

# Number 4
# ANSWER: The qqnorms of HW and final both fall above the qqnorm plot
#  for the standard normal distribution, while the qqnorm of
#  the lognormal distribution falls below the qqnorm of the 
#  standard normal dist. The skewness of the lognormal dist
#  is 4.04 and the skewness and thus right-skewed, while the 
#  HW and final samples are left-skewed with negative skewness.

# Number 5
x = (-30:30)/10
dens = dnorm(x)
dens_t = dt(x, df = 4)
dev.off()
plot(x, dens)
lines(x, dens_t)

# ANSWER: The tails of the student t distribution are taller than the tails of the 
#    normal distribution.

# Number 6
hist(dens)
hist(dens_t)
qqnorm(dens_t, main="qqnorm for t-dist with df=4")

# ANSWER: I see an S shape, but it looks flipped when compared to the plot
#  on page 175 in devore's book. It is heavy-tailed because the density curve declines less rapidly
#   towards the tails than the density plot of the standard normal distribution.
# 
###########################################





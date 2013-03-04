
# Author: Ted Chang (modified by M. Veeraraghavan, further modified by Ala Abdelbaki)
# Date: Sept. 18, 2012

# The dataset contains lottery numbers that were drawn for the Vietnam draft.
# This first lottery applied to persons born between Jan 1, 1944 and December 31, 1950.  
# Each registrant received a lottery sequence number based upon their date of birth.
# Because 1944 and 1948 were leap years, Feb. 29 was necessary.
# It was expected that numbers below 125 would be drafted.  

# The first draft lottery was unfair in the sense that later 
# birthdays (in a calendar year) tended to receive early draft
# sequence numbers. This resulted in young men with later
# birthdays having a higher chance of being drafted. A plot of 
# birthday versus sequence number was smoothed using a running mean 
# smoother and this tendency was clearly observed. This data set is 
# revisited later in the semester when a Monte Carlo hypothesis
# test is used to show that the first draft lottery is defective. 
# The same techniques applied to the second draft lottery failed to reject 
# a null hypothesis of fairness. It should be noted that the defects in 
# the first draft lottery were a subject of Congressional hearings,
# resulting in a redesign of the lottery.

# Two methods are tried: smoothing operations (running mean, smooth.spline)
# and hypothesis testing on Spearman rank correlation. Spearman rank correlation
# will show if a function is monotonically increasing or decreasing even if nonlinear
# while Pearson correlation shows r = 1 if linear. Since lot vs. date is nonlinear
# this is the second technique used.

lot<-read.table("lottery1.txt")
lot

# lottery1.txt data already has multiple columns and rows.
# 31 days (rows) by 12 months (columns)
# last row has NA for months that do not have Day 31.
# Also Feb. has NA for 30.
# R book, page 20;  Not available (missing value)

# At this point, type

lot[,1] # this yields the first column
lot[1,] # this yields the first row

# convert the lot data set into a matrix


# R book, page 70 and page 138; convert to matrix
lot<-as.matrix(lot)
lot		

# Note: lot now has row numbers

# as.matrix is a generic function. The method for data frames will 
# return a character matrix if there is any non-(numeric/logical/complex)
# column, applying format to non-character columns.
# Otherwise, the usual coercion hierarchy 
# (logical < integer < double < complex) will be used, e.g.,
# all-logical data frames will be coerced to a logical matrix, 
# mixed logical-integer will give a integer matrix, etc.

# When coercing a vector, it produces a one-column matrix.

lot<-as.vector(lot)
lot		

# To test that it is a vector, try
lot[1,]
# Answer
# Error in lot[1, ] : incorrect number of dimensions
lot[,1]
# Answer
# Error in lot[, 1] : incorrect number of dimensions
lot[32]
# Answer
# [1] 86 -- Which is the number assigned to Feb. 1.


# Note: lot has now 372 consecutive entries; the first 31 are 
# the lottery sequence numbers for Jan 1-31 
# then 29 entries for Feb 1-29, 
# then NA in entries 61 and 62 representing the nonexistent 
# dates Feb 30-31, etc.

# R book, page 21, 
# Returns a vector with TRUE for all entries that are not available
is.na(lot)

# Notice entries 61 and 62 of is.na(lot) are true because 
# these entries of lot are NA

lot<-lot[!is.na(lot)]		

# !is.na(lot) is the negation of is.na(lot)
lot					

# All NA entries are removed: 
# Feb 30 and 31, April 31, June 31, Sept. 31 and Nov. 31
# therefore 372-6 = 366

# 366 entries representing, in order, lottery sequence numbers 
# for Jan 1 - Dec 31

date<-1:366
plot(date,lot)
cor(date,lot)
# Answer:
# [1] -0.2256861

fit<-lm(lot~date)
abline(fit)

# R book, page 92
# Why is this linear model not a good idea? Because 
# we can clearly see from the scatter plot that the relationship
# is not linear.

# It appears that later dates do have a tendency to get lower 
# lottery numbers. The tools of correlation and least squares line 
# fitting are designed to detect, whether there exists a linear relationships 
# between x (date) and y (lot). Below, we explore methods to detect 
# nonlinear relationships. Actually the running mean (and spline smoothers) are
# linear smoothers per segment, but as a while these smoothers do not fit straight lines to all
# the data, they model nonlinear relationships.
# The model is y = f(x) + random error
# Smoothing techniques try to find a suitable candidate for the true f(x).
# The simplest such method is the running mean.
# For a running mean of length 2r+1, the i-th observation yi is 
# replaced by the mean of the 2r+1 observations 
# yi-r, yi-r+1, ..., yi, ..., yi+r that is the 2r+1 observations 
# closest to yi. Modifications have to be made for observations 
# close to the end because, for example, there is no y0 or y367.

par(mfrow=c(2,2))

# R book, page 14, repeated numbers
rmean<-rep(NA,366)

# Using r of 1 - which means except for the ends, mean of three numbers
# is being obtained, e.g. if i =3, mean(lot[2:4]) is determined.

# mean(lot[2]:lot[4]) is just the mean of lot[2] and lot[4]
# So use mean(lot[2:4]) to get the mean of lot[2], lot[3], lot[4]

for (i in 1:366) rmean[i]<-mean(lot[max(1,i-1):min(366,i+1)])

# To verify operation printout rmean, and test rmean[3] as an example
# (159+251+215)/3 = 208.33

rmean[3]
# Answer:
# [1] 208.3333
lot[2:4]
# Answer: 
# [1] 159 251 215

# scatter plot vs. date and lot with lines plot between date and rmean
plot(date,lot)
lines(date,rmean,lwd=2)	# lwd=2 makes a thicker line for better visibility
title("running mean length 3")

# repeat above with r = 5 in the smoothing operation
for (i in 1:366) rmean[i]<-mean(lot[(max(1,i-5):min(366,i+5))])
plot(date,lot)
lines(date,rmean,lwd=2)
title("running mean length 11")

# repeat for r = 10 - which means 21 points considered in smoothing
for (i in 1:366) rmean[i]<-mean(lot[(max(1,i-10):min(366,i+10))])
plot(date,lot)
lines(date,rmean,lwd=2)
title("running mean length 21")

# repeat for r = 20 - which means 41 points considered in smoothing
for (i in 1:366) rmean[i]<-mean(lot[(max(1,i-20):min(366,i+20))])
plot(date,lot)
lines(date,rmean,lwd=2)
title("running mean length 41")

# why does running mean find the monotonic function underlying the noisy data?
# By taking the mean of three numbers and replacing the middle number by the mean
# values, we are trying to draw a straight line between the two edge points.
# Example: (1,1), (2,5) and (3,3). The mean value of the middle number makes it (2,3)
# thus lowering the middle point to lie along that straight line.

# Adopting the terminology of signal processing (an important 
# collection of techniques usually taught in electrical engineering 
# departments which is greatly concerned with problems of this sort), 
# f(x) is called the "signal" and the random error is called the "noise".
# The running means are attempting to estimate f(x), that is to recover
# the signal. We see that the running mean of length 3 is too jumpy,
# for this data set, to recover an informative estimated signal.
# As we increase the length of the running mean, the estimated signal
# becomes increasingly smooth. The length of the running 
# mean (3, 11, 21, or 41) is a "bandwidth" type parameter.
# All smoothing techniques have a bandwidth type parameter which 
# determines the amount of smoothing.  A wide bandwidth results in a
# smoother estimated signal. However a smooth estimated signal means
# that any high frequency signal (again adopting the signal processing lingo)
# has been lumped into the fitted noise.  As a general proposition, 
# high frequency signal is very difficult to distinguish from random noise.
# Although there are statistical methods for choosing a suitable bandwidth,
# often what one does is look at the fitted signals and pick one that
# "seems right" that is consistent with any knowledge one might have about 
# what the signal should look like in the physical situation at hand.
# Examining these plots we see (except in the running mean of length 3) a 
# clear tendency for lottery sequence number to decrease with date.
# The running mean has the disadvantage that all 2r+1 observations 
# yi-r, yi-r+1, ..., yi, ...,  yi+r contribute equally to the fitted 
# signal at time i.  R has several other fitting techniques which 
# assign higher weights to observations close to time i.  We will 
# explore one of them called smooth.spline.

# start smooth.spline fitting

par(mfrow=c(2,2))
plot(date,lot)

# R book, page 101, 
fit<-smooth.spline(lot~date)
lines(fit,lwd=2)
fit$df
# Answer:
# [1] 2.715323
title("default smooth.spline: df=2.72")

# smooth.spline has three parameters which can be used to control
# the amount of smoothing: df, spar, and lambda.  They are 
# equivalent in the sense that choosing any one of them determines
# the other two.  We will explore df ("degrees of freedom").
# Lower values of df creates greater smoothing, that is lower values
# of df correspond to a larger bandwidth.
# smooth.spline has a statistical technique, which it uses by default, 
# called "cross validation" to choose a value of df.
# We can see that the default degree of smoothing produces, 
# for this dataset, a very smoothed curve.
# We explore the result of specifying less smoothing, that is the 
# result of increasing df.

plot(date,lot)
fit<-smooth.spline(lot~date,df=10)
lines(fit,lwd=2)
title("smooth.spline with df=10")
plot(date,lot)
fit<-smooth.spline(lot~date,df=30)
lines(fit,lwd=2)
title("smooth.spline with df=30")
plot(date,lot)
fit<-smooth.spline(lot~date,df=50)
lines(fit,lwd=2)
title("smooth spline with df=50")

# Examing these plots, we see that the estimated signals with 
# df=10, 30, and 50 resemble greatly the running mean estimated 
# signals with lengths 41, 21, and 11 respectively.
# The smooth.spline signals are smoother than the corresponding
# running mean signals; this is an effect of the "tapering" of the 
# weights in smooth.spline away from the center (that is observations 
# far from the i-th observation count less in the fitted signal at time i).
# All 4 plots show a clear tendency for later dates to have lower lottery 
# numbers.

######### Start technique-learning digression ########

# Just as correlation measures the tendency of (x,y) 
# to lie on a straight line, we have a technique called 
# "rank correlation" to measure the tendency of y to increase 
# with x (positive rank correlation) or to decrease with 
# x (negative rank correlation).  The difference between 
# rank correlation and correlation is that the former does not 
# assume a linear relationship. In other words, if the relationship
# between two variables is not linear (see R book, page 85 for the plot
# of weight vs. height), we can rank the data and look for correlation.
# It is likely that the tallest person will be the heaviest.

# The following is the height and weight of 7 males
ht<-c(73,71,72,173/2.54,67,72,72) # the fourth term works to 68.11
wt<-c(210,190,185,141,155,170,195)

# R book, page 88
rht<-rank(ht)
rht
# Answer:
# [1] 7 3 5 2 1 5 5 

# The fifth guy is the shortest so he is ranked 1; 
# the first guy is the tallest and receives rank 7.
# Three guys are exactly 6 feet tall: they receive the average 
# 5 of the ranks 4, 5, and 6.

rwt<-rank(wt)
rwt
# Answer:
# [1] 7 5 4 1 2 3 6

rank.correlation<-cor(rht,rwt)
rank.correlation
# Answer:
# [1] 0.7783118

# The rank correlation of ht and wt is the usual correlation of 
# their ranks rht and rwt.  Rank correlation is also called 
# "Spearman rank correlation" and R can compute it directly 
# from the original variables wt and ht.

# R book, page 88
cor(wt,ht,method="spearman")

# Answer:
#[1] 0.7783118

# Try the kid.weights dataset from UsingR following R book, page 85
# plot(height,weight) and then try plot(rheight,rweight) and see that the latter
# is linear - where rheight is rank of height. See statistical techniques
# on how rho = 0 does not mean that there is no relationship, just not a linear
# relationship. Monotonic relationships (even if nonlinear)
# are captured by high values of Spearman correlation
# unlike Pearson's which is close to 1 only when the relationship is linear

# We can also do a test using rank correlation.  
# The null hypothesis is that wt and ht do not have a tendency 
# to jointly increase or to move in opposite directions versus the 
# alternative that wt and ht have a tendency to jointly increase.


cor.test(wt,ht,alternative="greater",method="spearman")

# greater indicates that the alternative hypothesis is that
# the association (correlation if method is Pearson, and rank correlation
# if the method is Spearman) is positive. 
# The value of the association measure under the null hypothesis is always 0.

# Answer:
#        Spearman's rank correlation rho
#
# data:  wt and ht 
# S = 12.4145, p-value = 0.01964
# alternative hypothesis: true rho is greater than 0 
# sample estimates:
#      rho 
#   0.7783118 

# Warning message:
# In cor.test.default(wt, ht, alternative = "greater", method = "spearman") :
#  Cannot compute exact p-values with ties

# End Answer

# Explanation: Since the p-value is 0.02, it appears that ht and wt do 
# tend to jointly increase. Since there are 3 guys with height 72 inches, 
# only approximate p-values were computed. 

######### End technique-learning digression ########

# For the lottery data, the ranks of both date and lot are themselves.
# Hence rank correlation and ordinary correlation coincide.

cor(date,lot)
# Answer:
# [1] -0.2256861

cor(date,lot,method="spearman")
# Answer:
# [1] -0.2256861

cor.test(date,lot,alternative="less",method="spearman")

# Answer:
#        Spearman's rank correlation rho
#
# data:  date and lot 
# S = 10015394, p-value = 6.874e-06
# alternative hypothesis: true rho is less than 0 
# sample estimates:
#      rho 
#   -0.2256861 

# Also try Pearson's cor.test and see that the statistic is different.
# We will study this later in the semester with hypothesis testing.

# End Answer

# Explanation: The evidence is overwhelming (p < .000007) 
# that this level of tendency for lottery number to decrease with 
# increasing date would not have arisen by pure chance if the capsules 
# had been properly mixed. 
# In congressional testimony it was established that the dates 
# were put into the barrel in order starting with Jan 1 and hence the 
# later dates started at the top of the barrel. The mixing process used 
# was insufficient to destroy the tendency of the late dates to be close 
# to the top. Mixing experts pointed out that it is quite difficult 
# to obtain a true mix and that the first draft lottery was 
# insufficiently mixed.

# In the second draft lottery two barrels were used.
# One barrel had the dates Jan 1 - Dec 31 and the second barrel
# had the lottery sequence numbers 1 to 365. Both barrels were mixed 
# mechanically for a day.  Then one capsule was drawn from each barrel.  
# We see, for example, that Feb. 3 received lottery sequence number 186.
# This means that in the draw which drew the Feb. 3 capsule from the 
# first barrel, the sequence number 186 was drawn from the second barrel.  
# We cannot tell from the data whether this was the first draw, the 
# 20th draw, etc.


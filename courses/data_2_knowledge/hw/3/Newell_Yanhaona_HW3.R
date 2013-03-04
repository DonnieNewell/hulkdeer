
# Course: APMA 3501/STAT 3080/APMA 6548
# Assignment number: FILL IN

# Work in teams of 2

# Authors: Donald Newell, Muhammad Yanhaona
# Date: Sep-26-2012

# Group Pledge: On our honor as students, we have neither given nor received aid on this assignment.

# Save this R file using the following naming convention (please note the .R extension):
# Lastname1_Lastname2_HW#.R 

# Save all graphs, if any, in a single PDF file using this naming convention:
# Lastname1_Lastname2_Graphs_HW#.pdf 
# Start by writing your names in the graphs file you create.

# Submit your R file and PDF file via Collab (one submission per team; either team member can submit)

# Use this format for assignment #1
# Separate problems with a string of many #### as shown below

# List problem number as a comment (see example below)

# Show R command (Not as a comment - because the TA will run your program)

# After each command for which there is an output in the R console
# first have a line saying # Answer. In the next line,
# cut-and-paste answer from R Console into the program below
# and place a # symbol in front of the answer
# as shown below so that this is not
# executed when the whole code for the problem is selected for the 
# Run operation.


##########################################

# Problem 1: Lottery analysis
lot<-read.table("lottery2.txt")
lot<-as.matrix(lot)
lot<-as.vector(lot)
lot<-lot[!is.na(lot)]
date <- 1:length(lot)

par(mfrow=c(2,2))
plot(date,lot)

# R book, page 101, 
fit<-smooth.spline(lot~date)
lines(fit,lwd=2)
fit$df
# Answer:
# [1] 2.57
title("default smooth.spline: df=2.57")

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

cor(date,lot)
# Answer:
# [1] 0.01424944

cor(date,lot,method="spearman")
# Answer:
# [1] 0.01424944

cor.test(date,lot,alternative="less",method="spearman")
# Answer:
#  Spearman's rank correlation rho
#
# data:  date and lot 
# S = 7988976, p-value = 0.607
# alternative hypothesis: true rho is less than 0 
# sample estimates:
#   rho 
# 0.01424944 
# End Answer

cor.test(date,lot,alternative="less",method="pearson")
# Answer:
# Pearson's product-moment correlation
# 
# data:  date and lot 
# t = 0.2715, df = 363, p-value = 0.6069
# alternative hypothesis: true correlation is less than 0 
# 95 percent confidence interval:
#  -1.000000  0.100363 
# sample estimates:
#        cor 
# 0.01424944 
# End Answer

# Explanation:
# We can see from the spline plots that the lottery number seems to 
# be evenly assigned to the birth dates without bias. 
# The p-values of > .6 confirm that this sample does not show
# a statistically significant aberration from the NULL hypothesis
##########################################

##########################################

# Problem 2: Verzani 2.45

library(UsingR)
par(mfrow=c(1,2))
orig_pay = exec.pay
mean_orig = mean(orig_pay)
med_orig = median(orig_pay)
hist(orig_pay)

log_pay = log(1 + exec.pay, 10)
mean_log = mean(log_pay)
med_log = median(log_pay)
hist(log_pay)

# Answer:
# The original data set is right-skewed and it is hard to do any comparison.
# After the log transformation, the second histogram becomes more meaningful.
# This is because there are several pay ranges that can be compared, as 
# opposed to having virtually all observations reside in a single range.
# mean(original) = 59.889
# mean(log) = 1.44
# median(original) = 27
# median(log) = 1.45
# In the original pay data, extremely large pay values cause the mean to be 
# significantly larger than the median value. Because of the log compression, 
# the higher values scaled down much more than the lower values.
##########################################

##########################################

# Problem 3: Verzani 3.14

attach(galton)
cor(parent, child, method = "pearson")
# Answer: [1] 0.4587624

cor(parent, child, method = "spearman")
# Answer: [1] 0.4251345
detach(galton)
##########################################


##########################################
# Problem #4 Script: Verzani 3.27

# loading the library containing age data
library(UsingR)

# attaching dataset 
attach(batting)

# setting the graphix configuration
dev.off()

# creating a linear model of RBI modeled by HR
res <- lm(RBI~HR)

# ploting a scatter plot
plot(HR, RBI)

# fitting the linear model in the scatter plot
abline(res)

# getting Mike's index from dataset
mikesIndex <- which(playerID == 'piazzmi01')

# retrieving Mike's original HR and RBI
mikesHR <- HR[mikesIndex]
mikesRBI <- RBI[mikesIndex]

# displaying read values
cat('RBI: ', mikesRBI, ' HR: ', mikesHR)

# predicing Mike's RBI from the linear model
predictedRBI <- predict(res, data.frame(HR=mikesHR))
predictedRBI

# calculating the residual for Mike
residual <- mikesRBI - predictedRBI
residual

# detaching dataset
detach(batting)

# Problem #4 Answer:

# Mike Piazza's RBI and HR
# RBI:  98  HR:  33

# Predicted RBI from the linear model
# 104.1099

# Residual (error term) for Mike
#-6.1099 

##########################################


##########################################

# Problem #5 Script: Verzani 3.33

# loading the library containing age data
library(UsingR)

# attaching dataset 
attach(mw.ages)

# setting the graphix configuration
dev.off()

# plot both males and females together in a line curve
plot(1:103, Male+Female, type='l')

# draw a super smoother line for males on the plot
lines(supsmu(1:103, Male))

# draw a super smoother line for females on the plot
lines(supsmu(1:103, Female))

# detach the dataset
detach(mw.ages)

# Problem #5 Answer:
# The town's population consists mostly of people under 18 and
# between 30 and 50 years old. The main group 'missing' is the 
# 20 - 30 year olds.

###########################################


##########################################
# Problem #6 Script: Verzani 4.11

# loading the library containing ewr dataset
library(UsingR)

# extracting actual taxi in/out time data
df <- ewr[,3:10]

# calculating means by column
colMeans(df)

# calculating means by row
rowMeans(df)

# Problem #6 Answer:

# Column means
#AA       CO       DL       HP       NW       TW       UA       US 
#17.83478 20.01957 16.63043 19.60435 15.79783 16.28043 17.69130 15.49348 

# Row means
#1       2       3       4       5       6       7       8       9      10      11      12 
#8.6375  8.6375  8.4875  9.8625  9.0250  9.7375  8.2875  8.7375  8.0250  8.2000  8.6500  7.9250 
#13      14      15      16      17      18      19      20      21      22      23      24 
#8.3250  8.3250  8.3750  8.9125  9.0500  8.5625  8.2375  8.2125  8.1000  8.1125  8.5875 22.2000 
#25      26      27      28      29      30      31      32      33      34      35      36 
#24.0250 26.3125 32.2000 27.8250 30.9375 27.1375 26.0000 23.3125 24.0375 23.9500 22.7250 23.5000 
#37      38      39      40      41      42      43      44      45      46 
#25.1750 26.2125 29.3875 34.3750 28.8250 28.8000 26.8500 23.3625 22.9625 24.1500

# Taking means by row is interesting as it shows average taxi in/out time across airline
# carriers for specific years and months.

###########################################
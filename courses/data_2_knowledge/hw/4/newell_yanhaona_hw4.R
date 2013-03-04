
# Course: APMA 3501/STAT 3080/APMA 6548
# Assignment number: 4

# Work in the assigned teams of 2 (same teams as for Homework 1)

# Authors: Donnie Newell, Muhammed Yanhaona
# Date: 12Oct2012

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

# Problem 1: 
library(UsingR)
help(Animals)
data(Animals)
attach(Animals)

# (i)
log_body = log(body)
log_brain = log(brain)
plot(log_body, log_brain)
fit = lm(log_brain~log_body)
abline(fit)
title('log transformed body vs. brain')

# (ii)
fit

# ANSWER:
# > fit
# 
# Call:
#   lm(formula = log_brain ~ log_body)
# 
# Coefficients:
#   (Intercept)     log_body  
# 2.555        0.496

# (iii)
SSE_fit = deviance(fit)
SSE_fit
SST = sum((log_brain-mean(log_brain))^2)
SST
coeff_determination = 1 - SSE_fit/SST
coeff_determination
# ANSWER:
# SSE
# [1] 60.98799
# SST
# [1] 155.427
# coefficient of determination
# [1] 0.6076101

detach(Animals)

##########################################
###########################################

# Problem 2: Devore 7.14
attach(stud.recs)
names(stud.recs)
t.test(sat.m, conf.level=0.90)
detach(stud.recs)
# ANSWER:
# 476.8953 to 494.9797
#  One Sample t-test
# 
# data:  sat.m 
# t = 88.9145, df = 159, p-value < 2.2e-16
# alternative hypothesis: true mean is not equal to 0 
# 90 percent confidence interval:
#   476.8953 494.9797 
# sample estimates:
#   mean of x 
# 485.9375 

##########################################
###########################################

# Problem 3: Devore 7.19
#  single sample average height 67.5 inches, s = 2.54
#  3,000 points, find 95% confidence interval for the mean
#  height of the data.
# using Example 7.4 as a template
sample_avg = 67.5
sample_s = 2.54
n = 4
alpha = .05
t_star = qt(1 - alpha / 2, df = n - 1)
t_star
SE = sample_s / sqrt(n)
SE
t_star_times_SE = t_star * SE
conf_interval_population = sample_avg + t_star_times_SE * c(-1 , 1)
conf_interval_population
# ANSWER:
# [1] 63.45829 71.54171

##########################################
###########################################

# Problem 4: Devore 10.4
attach(galton)
names(galton)
help(jitter)
jit_parent = jitter(parent, factor = 1)
jit_child = jitter(child, factor = 1)
plot(jit_parent, jit_child)
fit = lm(child~parent)
abline(fit)
title('child height as function of parent height')
cor(child, parent)
deviance(fit)

# ANSWER:
# This is a good candidate for jitter, because the values are discrete,
# which means that the plot doesn't show how many points fall in each
# height category. By jittering the values, you get a sense of the 
# number of observations in each category.

# there is a moderate correlation between the child height and parent height
# cor(child, parent) == [1] 0.4587624
# the RSS is large, indicating that the quality of fit is not great.
# deviance(fit) == [1] 4640.273

##########################################
###########################################

# Problem 5: Devore 10.10



# ANSWER:
#

##########################################
###########################################

# Problem 6: Devore 10.12

# (i)

# (ii)

# (iii)

# ANSWER:
#

##########################################







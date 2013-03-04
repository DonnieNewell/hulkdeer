# Course: APMA 6548
# Assignment number: 5

# Authors: Donald Newell, Muhammad Yanhaona, Matt Steiner
# Date: 10/24/2012

# Group Pledge: On our honor as students, we have neither given nor received aid on this assignment.

###################################################################
# Problem #1 : Simulations from a population distribution of Exponential

prob1 <- function()
{
   # Collects all code repeated for N=40 & N=50
   cltCriteria <- function(N)
   {
      #Creates a list of 500 vectors, each of size N from the exponential distribution
      samples <- list()
      for(i in 1:500){samples[[i]] = rexp(N,1)}
   
      #Finds the sample means and variances
      samplemeans <- unlist(lapply(samples, mean))
      samplevars <- unlist(lapply(samples, var))
   
      #Plots the Q-Q Norm to test normality
      qqnorm(samplemeans, main=paste("Normal QQ Plot for N=", N))
      cat("For N=", N, "\n")
      
      # Calculates the mean of the sample means, expected 1^-1=1
      cat("Mean of Sample Means:", mean(samplemeans), "\n")
      cat("  Expected: 1\n")
      
      # Calculates StDev of the sample means, expected = SampVar/sqrt(n), SampVar=1^-2
      cat("St. Deviation of Sample Means:", sd(samplemeans), "\n")
      cat("  Expected: ", 1/sqrt(N), "\n")
      
      # Calculates mean Var of the samples, expected var=df/df-2
      cat("Mean Var. of Samples:", mean(samplevars), "\n")
      cat("  Expected: ", N/(N-2), "\n")
      
      # Calculates the upper and lower confidence values for 95%
      tcrit<-qt(.975,N-1)
      sampleupper <- samplemeans+tcrit*sqrt(samplevars/N)
      samplelower <- samplemeans-tcrit*sqrt(samplevars/N)
      
      # Tests whether the confidence intervals contain the true mean 1^-1, and what proportion
      contained = ifelse(1 < sampleupper & 1 > samplelower, 1, 0)
      cat("Number of 95% Intervals containing true mean:", sum(contained)/500, "\n")
      cat("  Expected: ", 0.95 , "\n\n")
   }
   
   # Executes the script for N=40 & N=50, collects the graphs
   par(mfrow=c(1,2))
   cltCriteria(40)
   cltCriteria(50)
}

#Problem #1 Answer:
prob1()
#Plots Generated in PDF
#
# For N= 40 
# Mean of Sample Means: 0.9989308 
# Expected: 1
# St. Deviation of Sample Means: 0.1510185 
# Expected:  0.1581139 
# Mean Var. of Samples: 0.9845518 
# Expected:  1.052632 
# Number of 95% Intervals containing true mean: 0.938 
# Expected:  0.95 

# For N= 50 
# Mean of Sample Means: 1.003259 
# Expected: 1
# St. Deviation of Sample Means: 0.1391202 
# Expected:  0.1414214 
# Mean Var. of Samples: 1.020692 
# Expected:  1.041667 
# Number of 95% Intervals containing true mean: 0.948 
# Expected:  0.95 
#
# QQNorm plots of the sample means shows that both are approximately normal
# Both distributions pass the listed CLT tests
#  
# For this particular interation the CLT approximation is slightly better for N=50, however there
# is significant variations between runs and both are similar enough this is not true for every
# simulation, however the N=50 is more often closer.


###################################################################
# Problem #2 : Textbook Problem 5.27

prob2 <- function()
{
   p <- .5
   n <- 100

   #calculates the probability of 42 or less for both distributions
   binom <- pbinom(42,n,p)
   norm <-pnorm(42, mean=n*p, sd= sqrt(n*p*(1-p)))
   
   cat("Binomial:", binom, "  Normal Approx.:", norm)
}

#Problem #2 Answer:
prob2()
# Binomial: 0.06660531   Normal Approx.: 0.05479929
# This is a fair but first order approximation


###################################################################
# Problem #3 : Textbook Problem 6.1

prob3 <- function()
{
   par(mfrow=c(1,2))
   qqnorm(rbinom(1000, 100, .02), main ="QQ Norm : p = 0.02")
   qqnorm(rbinom(1000, 100, .2), main ="QQ Norm : p = 0.2")
}

#Problem #3 Answer:
prob3()
# Plots generated in PDF
# The P=0.02 distribution is hard to discern as the number of successes in a sample of
# n=100 is very low, while the P=0.2 is clearly approximating a normal distribution.
# If n=1000 however it is easy to see that the P=0.02 distribution grows to be
# approximately normal in character


###################################################################
# Problem #4 : Textbook Problem 7.6

prob4 <- function()
{
   p =.05
   n= 100
   tcrit <-qnorm(.975)
   sterror <- sqrt(p*(1-p)/n)
   
   cat('A 95% confidence interval is between', p-tcrit*sterror, 'and', p+tcrit*sterror)
}

#Problem #4 Answer:
prob4()
# A 95% confidence interval is between 0.007283575 and 0.09271642
# This does not include the value of p = 1/10 = 0.1


###################################################################
# Problem #5 : Textbook Problem 7.10

prob5 <- function()
{
   p =.45
   n1 = 250
   n2 = 1000
   tcrit <-qnorm(.975)
   sterror1 <- sqrt(p*(1-p)/n1)
   sterror2 <- sqrt(p*(1-p)/n2)
   
   cat("For n =", n1, " A 95% confidence interval is between", p-tcrit*sterror1, "and", p+tcrit*sterror1, "\n")
   cat("For n =", n2, " A 95% confidence interval is between", p-tcrit*sterror2, "and", p+tcrit*sterror2)
}

#Problem #5 Answer:
prob5()
# For n = 250  A 95% confidence interval is between 0.3883312 and 0.5116688 
# For n = 1000  A 95% confidence interval is between 0.4191656 and 0.4808344
# The second interval is not four times smaller, it is sqrt(1000/250)=2 times smaller


###################################################################
# Problem #6 : Textbook Problem 7.12

prob6 <- function()
{
   #Code Directly from book to run simulation
   m = 50; n = 20; p = .5;
   alpha = .10; zstar = qnorm(1-alpha/2);
   phat = rbinom(m,n,p)/n
   SE = sqrt(phat*(1-phat)/n)
   cat("Propotion of intervals containing p :", sum(phat - zstar*SE < p & p < phat +zstar*SE)/m)
   par(mfrow=c(1,1))
   matplot(rbind(phat - zstar*SE, phat + zstar*SE),rbind(1:m,1:m),type="l",lty=1)
   abline(v=p)
}

#Problem #6 Answer:
prob6()
# Propotion of intervals containing p : 0.92
# This number will change with each simulation but is distributed around .90, (1-alpha)



###################################################################
# Problem #7 : Problem on distributions

# R provides four functions for each distribution, d, p, q, r
# To learn the usage of these four, since we know the normal distribution,
# let's start with normal. Notice that you are using all four (d, p, q, r) in
# answering the first five questions. 

# What is the probability that a standard normal distributed
# random variable X is negative? [Type in an R command even though you
# readily know the answer; the point is to learn the command for cases
# where you already know the answer]

pnorm(0, mean = 0, sd = 1)
#Answer:
#[1] 0.5


# What is the probability that a standard normal distributed
# random variable X, whose mean 10 and variance 4, is less than 10?
# (first find standard deviation from the variance)

pnorm(10, mean = 10, sd = sqrt(4))
#Answer:
#[1] 0.5


# Compare the probability density function values 
# for a standard normal random variable X for quantiles of +1 and -1

dnorm(1, mean = 0, sd = 1)
dnorm(-1, mean = 0, sd = 1)
#Answer:
#[1] 0.2419707
#[1] 0.2419707
#Values are the same, as distribution is centered on 0


# What is the 0.5th quantile value of a standard normal random variable?
# This question is asking for the smallest value x 
# such that F(x) = 0.5, where F is the distribution function.

qnorm(.5, mean = 0, sd = 1)
#Answer:
#[1] 0

# Generate a random sample of size 10 from a population that has
# a standard normal distribution

rnorm(10, mean = 0, sd = 1)
#Answer:
#[1] -0.82874691 -0.17149837 -1.04384527 -0.16837615  0.52059692 -1.02702689 -0.64273797 -0.01423997 -2.03831925  0.52731867


# Now on to other distributions:

# What is the probability that a Binomal random variable X with parameters
# size of 10 and probability of 0.4 takes on the value 8?

dbinom(8, size=10, prob=.4)
#Answer:
#[1] 0.01061683

# What is the probability that the same Binomial random variable from
# the previous question takes on the values in the range [4,8]?
# Answer this question using both pbinom and dbinom

pbinom(8, size=10, prob=.4)-pbinom(3, size=10, prob=.4)
sum(dbinom(4:8, size=10, prob=.4))
#Answer:
# [1] 0.6160417
# [1] 0.6160417

# Compare the 0.75th quantile values of a gamma distributed random variable
# with three values of the scale parameter: 0.5, 1, and 2. Use
# a value of 2 for the shape parameter for all three cases.

qgamma(.75, shape=2, scale=.5)
qgamma(.75, shape=2, scale=1)
qgamma(.75, shape=2, scale=2)
#Answer:
#[1] 1.346317
#[1] 2.692635
#[1] 5.385269
# Increasing the scale stretches the 75th quantile to larger values

# Generate a random sample of size 10 for a 
# Poisson distributed random variable with
# parameter 2. Is this a discrete or continuous random variable?

rpois(10, lambda=2)
#Answer:
#[1] 0 4 7 1 1 2 2 3 3 1
# This is a discrete random variable

# Finally, a practical use case

# A consumer is trying to decide between two long-distance 
# calling plans. The first one charges a flat rate of 10 cents
# per minute, whereas the second one charges a flat rate of 
# 99 cents per call for calls up to 20 minutes in duration
# and then charges 10 cents for each additional minute exceeding 20.
# Assume that calls lasting a noninteger number of minutes are
# rounded up to the higher integer value, e.g., a call lasting 20 minutes
# and 10 seconds will be charged as a 21 minute call.
# Assume that the call durations (after the rounding up) are distributed
# according to a Poisson distribution with parameter lambda.
# Neglect the probability of calls 30 minutes or longer.

# Which plan is better if the expected call duration is 10 minutes?
# One way is to compare mean charge/call.

# Generates the probability density function for 1 to 30 min, with an average call length of 10min
density <- dpois(1:30, 10)

# Generates the pricing for each plan
plan1 <- seq(10, 300, 10)
plan2 <- c(rep(99,20) , seq(109, 199, 10))

# Finds the expectation values for each plan
cat("Expectation value of Plan 1 :" , sum(density*plan1), "\n")
cat("Expectation value of Plan 2 :" , sum(density*plan2))

# Answer:
# Expectation value of Plan 1 :  99.99997
# Expectation value of Plan 2 :  99.02327
#
# The second plan would be cheaper if the expected average call length, with a poisson distribution, is 10 min
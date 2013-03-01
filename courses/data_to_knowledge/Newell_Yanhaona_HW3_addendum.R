
# Course: APMA 3501/STAT 3080/APMA 6548
# Assignment number: 3

# Work in teams of 2

# Authors: Donald Newell, Muhammad Nur Yanhaona
# Date: Sep-25-2012

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



###########################################

# Problem #1 Script:

# function for computing tri-cube weights
tri_cube <- function(x1, x2, h) {
  t <- (x1  - x2)/h
  if (abs(t) < 1) {
    return((1 - abs(t)^3)^3)  
  } else {
    return(0)
  }
}

# function that computes filtered means
my_filter <- function(data, n, weighting) {
  
  observations = length(data)
  fmean <- rep(NA,observations)
  
  # compute running mean
  if (as.character(weighting) == as.character("unweighted")) {
    for (i in 1:observations) {
      fmean[i] <- mean(lot[max(1, i-n) : min(observations, i+n)])
    }

  # compute tri-cube weight mean
  } else if (as.character(weighting) == as.character("weighted")) {
    
    # calculate the smoothing parameter
    f <- 0.5    
    r <- as.integer(round((2*n + 1)*f))
    
    # compute the mutual distances between independent variable observations,
    # which is in our case date (1,2,3,4, ..., 366)
    diffMatrix = matrix(rep(0, observations*observations), observations)
    for (i in 1:observations) {
      for (j in 1:observations) {
        diffMatrix[i,j] <- abs(i - j)
      }
    }
    
    # calculate values for h
    hvector <- rep(NA,observations)
    for (i in 1:observations) {
      sortedDiffs <- sort(diffMatrix[i,])
      hvector[i] <- sortedDiffs[r]
    }
    
    # calculate weights
    weightMatrix = matrix(rep(0, observations*observations), observations)
    for (i in 1:observations) {
      for (j in 1:observations) {
        weightMatrix[i,j] <- tri_cube(j, i, hvector[i])
      }
    }
    
    # calculate filtered means
    for (i in 1:observations) {
      filterPointSum <- 0
      for (j in 1:observations) {
        filterPointSum <- filterPointSum + weightMatrix[i,j] * data[j] 
      }
      fmean[i] <- filterPointSum / sum(weightMatrix[i,]) 
    }
  
  # weighting type did not match; showing error  
  } else {
    print("Unrecognized weighting input.")
  }
  
  # return filtered data
  return(fmean)
}
#####################################################
# MAIN SCRIPT

# read and process lottery data 
lot<-read.table("lottery1.txt")
lot<-as.matrix(lot)
lot<-as.vector(lot)
lot<-lot[!is.na(lot)]

# set the display
par(mfrow=c(2,2))

# compute and plot the running mean
fmean <- my_filter(lot, 5, "unweighted")
date<-1:length(lot)
plot(date,lot)
lines(date,fmean,lwd=2)
title("computed running mean")

# run the code from class
rmean<-rep(NA,length(lot))
for (i in 1:length(lot)) rmean[i]<-mean(lot[(max(1,i-5):min(length(lot),i+5))])
plot(date,lot)
lines(date,rmean,lwd=2)
title("running mean from class")

tcmean <- my_filter(lot, 5, "weighted")
plot(date,lot)
lines(date,tcmean,lwd=2)
title("computed tri-cube weight mean")

# Problem #1 Answer:
##########################################

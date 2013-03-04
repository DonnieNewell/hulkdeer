
# Course: APMA 3501/STAT 3080/APMA 6548
# Assignment number: 3 Grad Addendum

# Work in teams of 2

# Authors: Donnie Newell(den4gr), Muhammed Yanhaona
# Date: 24Sep2012

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
my_filter <- function(data, n, weighting) {
  data_length = length(data)
  coeff = rep(1, 2 * n + 1)
  filtered = rep(NA, data_length)
  if (weighting == "weighted") {
    
  } else if (weighting == "unweighted") {
    for (i in 1 : data_length) {
      samples = lot[(max(1, i - n) : min(data_length, i + n))]
      filtered[i] = mean( samples[:] * coeff[:])
    }
  }
  return(filtered)
}


#Problem #1 Answer:




##########################################
# Main Script

n = 8
radius = 3
lot<-read.table("lottery1.txt")
lot

lot<-as.matrix(lot)
lot

lot<-as.vector(lot)
lot  

lot<-lot[!is.na(lot)]
lot
rmean = my_filter(lot, n, "unweighted")
date<-1:length(lot)

par(mfrow=c(1,2))

plot(date,lot)
lines(date,rmean,lwd=2)
title(paste("running mean length ", radius * 2 + 1))

##########################################

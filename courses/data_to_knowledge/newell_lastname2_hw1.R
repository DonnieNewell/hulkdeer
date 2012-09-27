
# Course: APMA 3501/STAT 3080/APMA 6548
# Assignment number: 1

# Work in teams of 2

# Authors: Donnie Newell(den4gr)
# Date: 03Sep2012

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
fib_index = -1
while (fib_index <= 0) {
	cat("Enter the position of the fibonacci # to calculate: \n")
	fib_index = scan(n = 1)
}
fib_number = (((1 + sqrt(5)) / 2) ^ fib_index - ((1 - sqrt(5)) / 2) ^ fib_index) / sqrt(5)
cat('The fibonacci number at position ', fib_index, ' is ', fib_number, '.\n')

#Problem #1 Answer:
# The fibonacci number at position  1  is  1 .
# The fibonacci number at position  2  is  1 .
# The fibonacci number at position  4  is  3 .
# The fibonacci number at position  5  is  5 .

##########################################

# Problem #2 Script:
r = -1
n = -1
while (r < 1 || n < 1) {
	cat("Enter 2 non-negative integers: \n");
	r = scan(n = 1)
	n = scan(n = 1)
}
if (r > n) {
	temp = r
	r = n
	n = temp
}
n_fac = n
r_fac = r
n_minus_r_fac = n - r

# calculate n!
if (n_fac == 0) {
	n_fac = 1
} else if (n_fac > 1) {
	temp = n_fac - 1
	while (temp > 0) {
		n_fac = n_fac * temp
		temp = temp - 1
	}
}

# calculate r!
if (r_fac == 0) {
	r_fac = 1
} else if (r_fac > 1) {
	temp = r_fac - 1
	while (temp > 0) {
		r_fac = r_fac * temp
		temp = temp - 1
	}
}

# calculate (n-r)!
if (n_minus_r_fac == 0) {
	n_minus_r_fac = 1
} else if (n_minus_r_fac > 1) {
	temp = n_minus_r_fac - 1
	while (temp > 0) {
		n_minus_r_fac = n_minus_r_fac * temp
		temp = temp - 1
	}
}

# calculate n choose r
n_choose_r = n_fac / (r_fac * n_minus_r_fac)
cat(n, ' choose ', r, ' = ', n_choose_r, '.\n');

#Problem #2 Answer:
# 6  choose  2  =  15 .

###########################################

# Problem #3 Script:

#Define function:
factorial <- function(num) {
	n_factorial = num
	if (n_factorial == 0) {
		n_factorial = 1
	} else if (n_factorial > 1) {
		temp = n_factorial - 1
		while (temp > 0) {
			n_factorial = n_factorial * temp
			temp = temp - 1
		}
	}
	return(n_factorial)
}

n_ch_r <- function(n_num, r_num) {
	n_fac = factorial(n_num)
	r_fac = factorial(r_num)
	n_minus_r_fac = factorial(n_num - r_num)

	# calculate n choose r
	n_c_r = n_fac / (r_fac * n_minus_r_fac)
	return(n_c_r)
}

#Main script:
cat("Enter 2 non-negative integers: \n");
r = scan(n = 1)
n = scan(n = 1)

while (r < 0 || n < 0) {
	cat("Error: numbers must be non-negative. \n");
	cat("Enter 2 non-negative integers: \n");
	r = scan(n = 1)
	n = scan(n = 1)
}

if (r > n) {
	temp = r
	r = n
	n = temp
}

result = n_ch_r(n_num = n, r_num = r)
cat(n, ' choose ', r, ' = ', result, '.\n');




#Problem #3 Answer:
# 3  choose  2  =  3 .

###########################################
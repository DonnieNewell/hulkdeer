
# Course: APMA 3501/STAT 3080/APMA 6548
# Assignment number: 2

# Work in teams of 2

# Authors: Donald Newell, Muhammad Nur Yanhaona
# Date: Sep-16-2012

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

statGenerator <- function(dist) {
  total = sum(dist)
  observations = length(dist)
  avg = total / observations
  cat("computed mean: ", avg)
  
  sorted_dist = sort(dist)
  if (observations %% 2 == 0) {
    middle_point = observations / 2
    median = (sorted_dist[middle_point] + sorted_dist[middle_point + 1]) / 2
    cat(", median: ", median)
  } else {
    middle_point = (observations + 1) / 2
    median = sorted_dist[middle_point]
    cat(", median: ", median)
  }
  
  diff_fun <- function(x, mean) (x - mean)^2
  squared_diffs = sapply(dist, diff_fun, avg, simplify=TRUE)
  variance = sum(squared_diffs) / (observations - 1)
  standard_deviation = sqrt(variance)
  cat(", standard deviation: ", standard_deviation)
  
  trimIndexStart = observations * 0.2 + 1
  trimIndexEnd = observations * 0.8
  trimmedSum = 0
  trimmedObservations = 0
  for (i in 1:observations) {
    if (i >= trimIndexStart && i <= trimIndexEnd) {
      trimmedSum = trimmedSum + sorted_dist[i]
      trimmedObservations = trimmedObservations + 1
    }
  }  
  trimmed_avg = trimmedSum / trimmedObservations
  cat(", trimmed mean: ", trimmed_avg)
}

#Problem #1 Answer:

##########################################

# Problem #2 Script:

print("Enter the number of observations")
observation_count = scan(n=1)

generateData <- function(observation_count) {
  validInput = FALSE
  while (!validInput) {
    validInput = TRUE
    print("Select a distribution type (exponential, chi-squared, or uniform)")
    type = scan(n=1, what='character')
    type
    if (as.character(type) == as.character("uniform")) {
      print("Generating a sample of uniform distribution")
      return(runif(observation_count))
    } else if (as.character(type) == as.character("chi-squared")) {
      print("Generating a sample of chi-squared distribution with degrees of freedom 2")
      return(rchisq(observation_count, df=2))
    } else if (as.character(type) == as.character("exponential")) {
      print("Generating a sample of exponential distribution")
      return(rexp(observation_count))
    } else {
      validInput = FALSE
      print("Unsupported distribution type. Please try again.")
    }
  }
}

dist <- generateData(observation_count)
print("Sample distribution:")
print(dist)

statGenerator(dist)
cat("\nactual mean: ", mean(dist), " median: ", median(dist), " standard deviation: ", sd(dist), " trimmed mean: ", mean(dist, trim=.2))

par(mfcol=c(2,1))
hist(dist)
lines(density(dist))
boxplot(dist)

#Problem #2 Answer:
# > source('~/.active-rstudio-document')
# [1] "Enter the number of observations"
# 1: 100
# Read 1 item
# [1] "Select a distribution type (exponential, chi-squared, or uniform)"
# 1: exponential
# Read 1 item
# [1] "Generating a sample of exponential distribution"
# [1] "Sample distribution:"
# [1] 0.05962571 1.37307844 0.48514329 0.51214155 1.23090578 0.23421502 0.32096232 0.44975525 2.46053403
# [10] 0.43329383 0.38958086 0.37448160 1.19786344 1.42667879 3.07904996 3.13943698 0.02387986 0.42100548
# [19] 0.22174245 0.19967156 1.51903062 0.85131467 0.48234302 0.77459636 0.26290009 0.82312656 0.71414245
# [28] 0.70963946 1.74575359 0.06652136 0.69532679 0.57571530 0.37958852 0.47572008 1.52332154 0.26477409
# [37] 0.14984131 1.41693220 0.87640341 0.03106232 0.72302971 0.99298735 0.15809313 1.03276076 1.16768032
# [46] 3.23054480 0.62367405 0.73703220 1.58464269 0.60034499 0.13180890 2.16652629 1.12659504 0.00980772
# [55] 0.01808496 0.11310519 0.27716183 1.78685785 1.49811937 0.08103475 0.62567205 1.84051039 0.51891244
# [64] 0.74395936 1.18378899 1.93842983 0.33836368 0.21501227 0.94519949 0.32911056 1.33877174 0.94305471
# [73] 0.40217256 1.93040035 0.42890677 2.78232035 0.26008604 0.21927577 0.82859181 0.39039222 0.58847942
# [82] 0.16548183 0.54618586 0.15472675 0.47264341 0.08128181 1.22775631 0.53616793 1.85488138 1.90995442
# [91] 2.31649315 1.23607561 1.52911769 1.76736828 0.68881270 0.78819731 4.06399039 1.72759322 0.75002221
# [100] 0.92549632
# computed mean:  0.9196465, median:  0.7024831, standard deviation:  0.8079058, trimmed mean:  0.7498761
# actual mean:  0.9196465  median:  0.7024831  standard deviation:  0.8079058  trimmed mean:  0.7498761


# > source('~/.active-rstudio-document')
# [1] "Enter the number of observations"
# 1: 100
# Read 1 item
# [1] "Select a distribution type (exponential, chi-squared, or uniform)"
# 1: chi-squared
# Read 1 item
# [1] "Generating a sample of chi-squared distribution with degrees of freedom 2"
# [1] "Sample distribution:"
# [1] 3.552765973 2.849904247 0.411205150 0.843450066 4.414996142 0.526892856 1.488634046 8.435371011 0.013591379
# [10] 1.532725815 2.377644073 2.923955636 1.957356861 1.805872164 0.191129351 1.030873423 6.995588425 3.921595702
# [19] 1.691492257 0.479634159 0.526654601 0.051082719 4.245089643 1.155739095 0.146393917 4.660133684 3.372594930
# [28] 3.789407164 4.625410824 1.437771992 0.306936872 0.682693577 0.797624194 1.110233348 1.481922071 0.159311487
# [37] 0.397550212 0.156398831 2.461734360 3.593359162 0.189196798 4.279823356 1.218300004 4.303139270 0.257339623
# [46] 4.466245334 0.030222803 3.581062417 5.570336257 0.489929248 3.522009468 0.002885535 0.016181361 1.319700879
# [55] 0.028850492 1.459486907 2.308534186 1.923945318 0.068840255 1.285495092 1.777120697 4.171199634 1.486194423
# [64] 6.087273145 0.158199638 0.257704816 4.981250069 0.832392955 5.394629346 1.025414243 0.070765466 0.589700481
# [73] 0.348261329 0.407059881 0.878399730 4.826775144 0.621398486 2.131659603 6.420046759 0.383597835 0.608540546
# [82] 0.896578861 5.117627950 3.399689483 1.902620317 0.122496813 4.309122427 0.349168064 0.136908917 1.768519148
# [91] 2.108136999 8.380168877 0.427412832 1.689970176 1.019904920 1.994962786 2.047624747 4.135216415 3.709928222
# [100] 2.478224388
# computed mean:  2.083761, median:  1.484058, standard deviation:  1.988379, trimmed mean:  1.664066
# actual mean:  2.083761  median:  1.484058  standard deviation:  1.988379  trimmed mean:  1.664066


# > source('~/.active-rstudio-document')
# [1] "Enter the number of observations"
# 1: 100
# Read 1 item
# [1] "Select a distribution type (exponential, chi-squared, or uniform)"
# 1: uniform
# Read 1 item
# [1] "Generating a sample of uniform distribution"
# [1] "Sample distribution:"
# [1] 0.492771479 0.610687600 0.968372248 0.128191249 0.834808979 0.001917158 0.755493803 0.261248993 0.967365522
# [10] 0.870290777 0.109453717 0.656685722 0.372024082 0.059842922 0.869137928 0.025893827 0.937888996 0.557890589
# [19] 0.789770657 0.266042263 0.904765821 0.861130692 0.049756435 0.916998438 0.993380349 0.173993838 0.073626645
# [28] 0.496268800 0.740805387 0.825414298 0.067121684 0.706584848 0.311896296 0.654478127 0.891406507 0.865686953
# [37] 0.425688813 0.632787582 0.967440982 0.100386693 0.265761063 0.751236147 0.697677547 0.357551815 0.679227074
# [46] 0.217759580 0.550162359 0.970302394 0.030017967 0.267462231 0.561047657 0.059835389 0.055520566 0.491734850
# [55] 0.228793819 0.578365665 0.838444037 0.463412991 0.332900160 0.793698959 0.784967049 0.407375113 0.942301935
# [64] 0.670654505 0.105517638 0.316260811 0.854436205 0.772356958 0.061676837 0.699008716 0.187347340 0.510110397
# [73] 0.119177735 0.651149143 0.437388131 0.148414496 0.878482067 0.050345915 0.210566980 0.829756783 0.792542834
# [82] 0.253180310 0.020006005 0.762528221 0.570288947 0.689403293 0.692137500 0.703507331 0.013299513 0.989459521
# [91] 0.656656304 0.424937486 0.363131319 0.046384294 0.590722454 0.130269577 0.998301989 0.679317808 0.964833712
# [100] 0.251355990
# computed mean:  0.5161587, median:  0.5656683, standard deviation:  0.3172859, trimmed mean:  0.5309533
# actual mean:  0.5161587  median:  0.5656683  standard deviation:  0.3172859  trimmed mean:  0.5309533


###########################################

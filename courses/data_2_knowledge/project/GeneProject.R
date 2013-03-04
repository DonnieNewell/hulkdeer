#Reads in the gene data, creates a list, with a vector for each gene with NA values removed
geneDataFrame <- read.csv("genes.csv")
numGenes <- length(geneDataFrame)
geneVectorList <- list()
for (currGene in 1 : numGenes) {
  geneVectorList[[currGene]] = geneDataFrame[,currGene][!is.na(geneDataFrame[,currGene])]
}

#Creates a list of vectors containing the differences in location between enzymes for each gene
geneDiffList <- list()
for (currGene in 1 : numGenes) {
   tempVector <- c(1 : (length(unlist(geneVectorList[currGene]))-1))
   for (j in 1 : length(unlist(geneVectorList[currGene])) - 1) {
      tempVector[j] <- unlist(geneVectorList[currGene])[j + 1] - unlist(geneVectorList[currGene])[j]
   }
   geneDiffList[[currGene]] <- tempVector
}
geneDiffsort <- sort(unlist(geneDiffList))

#Generates a pair of lists, allows ready comparison of one enzyme spacing with the following enzyme spacing
genePrevList <- list()
geneNextList <- list()
for(i in 1 : numGenes) {
   tempVectorPrev <- c(1 : (length(unlist(geneDiffList[i])) - 1))
   tempVectorNext <- c(1 : (length(unlist(geneDiffList[i])) - 1))
   for(j in 1 : (length(unlist(geneDiffList[i])) - 1)) {
      tempVectorPrev[j] <- unlist(geneDiffList[i])[j]
      tempVectorNext[j] <- unlist(geneDiffList[i])[j+1]
   }
   genePrevList[[i]] <- tempVectorPrev
   geneNextList[[i]] <- tempVectorNext
}

#Plots lines showing the trend of genes locations
plot(unlist(geneVectorList[1]), 1 : length(unlist(geneVectorList[1])), type="l", main="Fig.1: Enzyme Positions", xlab="Location on Gene", ylab="Enzyme Number")
for(i in 2 : numGenes) {
  points(unlist(geneVectorList[i]), 1 : length(unlist(geneVectorList[i])), type="l")
}

#Shows the length of the genes follows approximately a gamma distribution
len <- c(1:numGenes)
for(i in 1:numGenes){len[i] <- length(unlist(geneVectorList[i]))}
hist(len, main="Fig.2: Enzyme Count", xlab="Enzyme Count", ylab="Number of Genes")
qqplot(len, rgamma(10000,shape=mean(len)^2/var(len),rate=mean(len)/var(len)), main="Fig.3: QQPlot: Enzyme Count & Gamma", xlab="Gene Length", ylab="Gamma Distribution")
abline(0,1)


#Plots lines showing trend lines normalized by length
plot(unlist(geneVectorList[1]), (1:length(unlist(geneVectorList[1])))/length(unlist(geneVectorList[1])), type="l", main="Fig.4 :Normalized Enzyme Positions", xlab="Location on Gene", ylab="Normalized Enzyme Count")
for(i in 2:numGenes){points(unlist(geneVectorList[i]), (1:length(unlist(geneVectorList[i])))/length(unlist(geneVectorList[i])), type="l")}

#Shows the enzyme spacing histograms of the first twelve of the genes, all have similar behavior
par(mfrow=c(3,4))
for(i in 1:12){hist(unlist(geneDiffList[i]), breaks=50, xlim=c(0,15),main="Fig.5: Inter-Enzyme Distance", xlab="Gene Distance", ylab="Number of Occurances")}

#Shows a histogram of all the inter-gene distances
dev.off()
hist(geneDiffsort, breaks=100, xlim=c(0,15),main="Fig.6: Inter-Enzyme Distance", xlab="Gene Distance", ylab="Number of Occurances")

#Plots the CDF of the enzyme spacings
plot(geneDiffsort, 1:length(geneDiffsort), main="Fig.7: CDF of Enzyme Spacing", xlab="Enzyme Spacing", ylab="Number of Occurances")
#Closely resembles the CDF for an exponential distribution

#Plots the CDF of an exponential from uniform distribution for comparison
plot(exp(sort(runif(4193, min=0.0001, max=10))), 1:4193, main="Fig.8: CDF of Exponential Distribution (Uniform Random)", xlab="Arbitrary", ylab="Number of Occurances")

#Plots the CDF of an exponential from a normal distribution for comparison
plot(exp(sort(rnorm(4193))), 1:4193, main="Fig.9: CDF of Exponential Distribution (Normal Random)", xlab="Arbitrary", ylab="Number of Occurances")

#Plots the CDF of the enzyme log(spacing)
plot(log(geneDiffsort),1:length(geneDiffsort),main="CDF of Enzyme Log(Spacing)", xlab="Enzyme Log(Spacing)", ylab="Number of Occurances")
#Closely resembles the CDF for a normal distribution

# Creates an Ideal normal set to compare too
normVals <- qnorm(seq(1/(length(geneDiffsort)+1),1-1/(length(geneDiffsort)+1), 1/(length(geneDiffsort)+1)))
plot(normVals,1:length(normVals), main="Ideal Normal Distribution", ylab="Number of Occurances")

#Superimposes Plots, shows deviation at low values
plot(log(geneDiffsort),1:length(geneDiffsort),main="Fig.10 :CDF of Enzyme Log(Spacing), Normal=blue", xlab="Enzyme Log(Spacing)", ylab="Number of Occurances")
points(normVals,1:length(normVals),col="blue")

#Produces a QQNorm plot of the log
qqnorm(log(geneDiffsort), main="Fig.11: QQNorm of Enzyme Log(Spacing)")
abline(0,1)
#Great Fit at high quantiles, missing additional term at low values

# Try correcting for low values with some simple inverse functions
par(mfrow=c(1,3))
qqnorm(log(geneDiffsort)-1/geneDiffsort, main="Fig.11: Normal QQ, 1/x mod")
abline(0,1)
qqnorm(log(geneDiffsort)-1/geneDiffsort^2,, main="Fig.11:Normal QQ, 1/x^2 mod")
abline(0,1)
qqnorm(log(geneDiffsort)-1/sqrt(geneDiffsort),, main="Fig.11:Normal QQ, 1/sqrt(x) mod")
abline(0,1)
# Inverse Square Root produces a good single slope line, need to scale correctly


# Models a fit around the log and inverse square root terms
bestFit <- lm(formula = normVals ~ I(log(geneDiffsort)) + I(1/sqrt(geneDiffsort)))
summary(bestFit)
# Near Perfect fit

#Residuals:
#   Min       1Q   Median       3Q      Max 
#-0.12418 -0.01137  0.01182  0.02128  0.52959 
#
#Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
#(Intercept)              1.234940   0.006884   179.4   <2e-16 ***
#   I(log(geneDiffsort))     0.593492   0.002898   204.8   <2e-16 ***
#   I(1/sqrt(geneDiffsort)) -1.389865   0.006432  -216.1   <2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
#
#Residual standard error: 0.04246 on 4190 degrees of freedom
#Multiple R-squared: 0.9982,   Adjusted R-squared: 0.9982 
#F-statistic: 1.156e+06 on 2 and 4190 DF,  p-value: < 2.2e-16 

#Plots the QQNorm of the new model
dev.off()
qqnorm(coef(bestFit)[1] + coef(bestFit)[2]*log(geneDiffsort)+coef(bestFit)[3]/sqrt(geneDiffsort), main="Fig.12: Normal QQ, Fitted Model")
abline(0,1)
#Model is an exceptional fit for the data to a normal distribution

#Cumulative distibution function
cdf <- function(i){return(pnorm(coef(bestFit)[1] + coef(bestFit)[2]*log(i)+coef(bestFit)[3]/sqrt(i)))}

#Plots distribution derived from cumulative distribution function against real data
probDens <- c(1:200)
for(i in 1:200){probDens[i] <-cdf(i/2)-cdf((i-1)/2)}
hist(geneDiffsort, breaks=100, xlim=c(0,15),main="Fig.13: Inter-Enzyme Distance", xlab="Gene Distance", ylab="Number of Occurances")
points((1:200)/2, 4139*probDens[1:200], col="blue")

#Testing for correlations between spacings (if previous spacing impacts next)
plot(unlist(genePrevList),unlist(geneNextList),main="Spacing versus Following Spacing", xlab="Enzyme Spacing", ylab="Next Enzyme Spacing")
pairedModel <- lm(formula = unlist(genePrevList) ~ unlist(geneNextList))
summary(pairedModel)$r.squared

#Tests the normal fitted values of the spacings on the following spacing
plot(coef(bestFit)[1] + coef(bestFit)[2]*log(unlist(genePrevList))+coef(bestFit)[3]/sqrt(unlist(genePrevList)),coef(bestFit)[1] + coef(bestFit)[2]*log(unlist(geneNextList))+coef(bestFit)[3]/sqrt(unlist(geneNextList)),xlab="Fitted Spacing Value", ylab="Fitted Value of Next Spacing", main="Fig.14: Fitted Spacing vs Following Fitted Spacing")
pairedFitModel <- lm(formula = I(coef(bestFit)[1] + coef(bestFit)[2]*log(unlist(genePrevList))+coef(bestFit)[3]/sqrt(unlist(genePrevList))) ~ I(coef(bestFit)[1] + coef(bestFit)[2]*log(unlist(geneNextList))+coef(bestFit)[3]/sqrt(unlist(geneNextList))))
summary(pairedFitModel)$r.squared

#Plots graphs from a sample of twelve showing all follow the same behavior
par(mfrow=c(3,4))
for(i in 1:12){plot(coef(bestFit)[1] + coef(bestFit)[2]*log(unlist(genePrevList[i]))+coef(bestFit)[3]/sqrt(unlist(genePrevList[i])),coef(bestFit)[1] + coef(bestFit)[2]*log(unlist(geneNextList[i]))+coef(bestFit)[3]/sqrt(unlist(geneNextList[i])),xlab="Fitted Spacing Value", ylab="Fitted Value of Next Spacing", main="Fig.15: Fitted Spacing vs Following")}

#Runs a simulation to create gene lengths from the distribution, should match gene lengths given
probDensHigh <- c(1:1000)
for(i in 1:1000){probDensHigh[i] <-cdf(i/10)-cdf((i-1)/10)}
geneLengthsSim <- c(1:77)
for(i in 1:1000)
{
   size=0
   count=0
   while(size<100)
   {
      size <- size + sample((1 : 1000)/10, 1, prob = probDensHigh)
      count <- count + 1
   }
   geneLengthsSim[i] <- count
}
dev.off()
hist(geneLengthsSim, main="Fig.16: Histogram of Gene Length Simulation", xlab="Gene Length")

#Creates a new enzyme spacing distribution, this time weighted by the gene length
geneWeightDiffList <- list()
for(i in 1 : numGenes) {
   tempVector <- c(1 : length(unlist(geneVectorList[i])) - 1)
   for(j in 1 : length(unlist(geneVectorList[i])) - 1) {
      tempVector[j] <- unlist(geneVectorList[i])[j + 1] - unlist(geneVectorList[i])[j]
   }
   geneWeightDiffList[[i]] <- tempVector*length(unlist(geneVectorList[i]))
}
geneWeightDiffListSort <- sort(unlist(geneWeightDiffList))

#Plots the new weighted spacing distribution
hist(geneWeightDiffListSort, breaks=100, xlim=c(0, 500), main="Fig.17: Histogram of Weighted Enzyme Spacings", xlab="Weighted Enzyme Spacing")

#Plots the log
qqnorm(log(geneWeightDiffListSort), main="Fig.18: QQNorm of the Weighted Enzyme Spacings", ylab="Weighted Enzyme Spacing")
abline(0, 1)

# Models a fit around the log of the weighted spacings
weightFit <- lm(formula = normVals ~ I(log(geneWeightDiffListSort)))
summary(weightFit)

#Residuals:
#   Min       1Q   Median       3Q      Max 
#-0.94121 -0.04058  0.01255  0.07188  0.08923 
#
#Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
#(Intercept)                    -5.433948   0.007171  -757.8   <2e-16 ***
#   I(log(geneWeightDiffListSort))  1.268554   0.001647   770.4   <2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
#
#Residual standard error: 0.08361 on 4191 degrees of freedom
#Multiple R-squared: 0.993,   Adjusted R-squared: 0.993 
#F-statistic: 5.935e+05 on 1 and 4191 DF,  p-value: < 2.2e-16

#Plots the fitted weighted spacing
qqnorm(coef(weightFit)[1] + coef(weightFit)[2] * log(geneWeightDiffListSort), main="Fig.19: Normal QQ, Fitted Weighted Model")
abline(0, 1)

#Creates the CDF function for the weighted
cdf2 <- function(i){ return(pnorm(coef(weightFit)[1] + coef(weightFit)[2] * log(i))) }

#Plots distribution derived from cumulative distribution function against weighted data
probDens2 <- c(1 : 25)
for(i in 1 : 25) { probDens2[i] <-cdf2(i * 20) - cdf2((i - 1) * 20) }
hist(geneWeightDiffListSort, breaks = 100, xlim = c(0, 500), main = "Fig.20: Weighted Inter-Enzyme Spacing", xlab = "Weighted Enzyme Spacing", ylab = "Number of Occurances")
points((1 : 25) * 20, 4139 * probDens2[1 : 25], col = "blue")


#Generates a pair of weighted lists, allows ready comparison of one enyme spacing with the following enzyme spacing
genePrevWeightList <- list()
geneNextWeightList <- list()
for(i in 1 : num_genes) {
   tempVectorPrev <- c(1 : (length(unlist(geneDiffList[i])) - 1))
   tempVectorNext <- c(1 : (length(unlist(geneDiffList[i])) - 1))
   for(j in 1 : (length(unlist(geneDiffList[i]))-1)) {
      tempVectorPrev[j] <- unlist(geneDiffList[i])[j]
      tempVectorNext[j] <- unlist(geneDiffList[i])[j+1]
   }
   genePrevWeightList[[i]] <- tempVectorPrev*length(unlist(geneVectorList[i]))
   geneNextWeightList[[i]] <- tempVectorNext*length(unlist(geneVectorList[i]))
}

#Shows there is no correlation between subsequent values in enzyme spacing
plot(coef(weightFit)[1] + coef(weightFit)[2] * log(unlist(genePrevWeightList)), coef(weightFit)[1] + coef(weightFit)[2] * log(unlist(geneNextWeightList)),xlab="Fitted Spacing Value", ylab="Fitted Value of Next Spacing", main="Fig.22: Weighted Fitted Spacing vs Following Fitted Spacing")
pairedFitModel2 <- lm(formula = I(coef(weightFit)[1] + coef(weightFit)[2] * log(unlist(genePrevWeightList)) ~ I(coef(weightFit)[1] + coef(weightFit)[2] * log(unlist(geneNextWeightList)))))
summary(pairedFitModel2)$r.squared

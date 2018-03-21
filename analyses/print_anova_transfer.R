# R -e 'install.packages("corrgram", repos="http://cran.us.r-project.org")'
#
# ========== Starts up script ==========

args = commandArgs(TRUE)
jbhi_input      = ifelse(length(args)>=1, args[1], "results.anova.transfer.txt")
anova_output    = ifelse(length(args)>=2, args[2], "transfer_anova_table.csv")
averages_output = ifelse(length(args)>=2, args[2], "transfer_averages_table.csv")

# Fetches all results
jbhi = read.table(jbhi_input, header=TRUE, sep=";")

# ========== Computes ANOVA of ISBI AUCs over dataset ==========

# Reformats the resutls table so that corresponding results
# on different datasets are side-by-side
jbhi0 = jbhi[jbhi$j==0, ]
jbhi1 = jbhi[jbhi$j==1, ]
jbhi2 = jbhi[jbhi$j==2, ]
jbhi3 = jbhi[jbhi$j==3, ]
jbhi4 = jbhi[jbhi$j==4, ]

stopifnot(nrow(jbhi)==256*5)
stopifnot(nrow(jbhi0)==256)
stopifnot(nrow(jbhi1)==256)
stopifnot(nrow(jbhi2)==256)
stopifnot(nrow(jbhi3)==256)
stopifnot(nrow(jbhi4)==256)

# ========== Performs ANOVA ==========

attach(jbhi)

logodds <- function(prob) {
        return(log10(prob/(1-prob))*10.0)
    }

fit = lm( logodds(isbi_auc) ~ (a + b + c + d + e + f + g + h + i + j + t)^3 )
model = anova(fit)

significance = 0.95
alpha = 1-significance
cat("Number of factors/combinations ", significance, "=", nrow(model), sep=" ")
cat("\n")
cat("Number of factors/combinations significant at", significance, "=", nrow(model[model$`Pr(>F)`<alpha,]), sep=" ")
cat("\n")

totalSq  = sum(model[, "Sum Sq"])
all_j = c("j", "a:j", "b:j", "c:j", "d:j", "e:j", "g:j", "i:j", "j:t", "a:b:j", "a:c:j", "a:d:j", "a:e:j", "a:g:j", "a:i:j", "a:j:t", "b:c:j", "b:d:j", "b:e:j", "b:g:j", "b:i:j", "b:j:t", "c:d:j", "c:e:j", "c:g:j", "c:i:j", "c:j:t", "d:e:j", "d:g:j", "d:i:j", "d:j:t", "e:g:j", "e:i:j", "e:j:t", "g:i:j", "g:j:t", "i:j:t")
totalSqJ = sum(model[all_j, "Sum Sq"])
residualsSq = model["Residuals", "Sum Sq"]
mainSq = totalSq - totalSqJ - residualsSq

model$FracSq = with(model, `Sum Sq`/totalSq)
model$FracMainSq = with(model, `Sum Sq`/mainSq)


# plot(residuals(fit)~fitted(fit))
# qqnorm(residuals(fit))
# qqline(residuals(fit))

all_factors = rownames(model)
single_factors = all_factors[nchar(all_factors)==1]
large_factors = rownames(model[model$FracMainSq>=0.01,])
important_factors = large_factors[nchar(large_factors)>1]
selectedFactors = c(single_factors, important_factors)
selectedColumns = c("Pr(>F)", "FracSq", "FracMainSq")

output = model[selectedFactors, selectedColumns]

write.table(output, sep="\t", dec=",", na="#NA", quote=FALSE, file=anova_output)

to_var <- function(varname) eval(parse(text=varname))

sink(averages_output)
cat("Factors", "Max_Treat", "Max_Average", "Min_Treat", "Min_Average", sep="\t")
cat("\n")
for(index in 1:(length(selectedFactors)-1)) {
  factors = selectedFactors[index]
  fac = strsplit(factors, ":")[[1]]
  agg = aggregate(isbi_auc, by=lapply(fac, to_var), FUN=mean, data=jbhi)
  maxval = as.numeric(agg[which.max(agg$x),])
  minval = as.numeric(agg[which.min(agg$x),])
  cat(factors)
  cat('\t"')
  cat(maxval[1:(length(maxval)-1)])
  cat('"\t')
  cat(maxval[length(maxval)])
  cat("\t")
  cat('\t"')
  cat(minval[1:(length(minval)-1)])
  cat('"\t')
  cat("\t")
  cat(minval[length(minval)])
  cat("\n")
}
sink()

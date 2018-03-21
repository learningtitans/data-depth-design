# R -e 'install.packages("corrgram", repos="http://cran.us.r-project.org")'
#
# ========== Starts up script ==========

args = commandArgs(TRUE)
jbhi_input      = ifelse(length(args)>=1, args[1], "all_results.anova.txt")
selection_input = ifelse(length(args)>=2, args[2], "selection.txt")
sort_dataset    = ifelse(length(args)>=3, args[2], "0")

sort_dataset = as.numeric(sort_dataset)

# Fetches all results
jbhi = read.table(jbhi_input, header=TRUE, sep=";")
selected_lines = read.table(selection_input, header=TRUE)

# ========== Seelects data ==========

# Reformats the resutls table so that corresponding results
# on different datasets are side-by-side
jbhi_sort = jbhi[jbhi$j==sort_dataset, ]

stopifnot(nrow(jbhi)==512*5)
stopifnot(nrow(jbhisel)==512)

# default => , by=c("a", "b", "c", "d", "e", "f", "g", "h", "i"), all.x=FALSE, all.y=FALSE
jbhisel = merge(x=jbhi_sort, y=selected_lines)
stopifnot(nrow(jbhisel)==nrow(selected_lines))

# ========== Performs ANOVA ==========

attach(jbhisel)

logodds <- function(prob) {
        return(log10(prob/(1-prob))*10.0)
    }

# Main effects only
fit = lm( logodds(isbi_auc) ~ a + b + c + d + e + f + g + h + i + j )
fit

model = anova(fit)
model


# R -e 'install.packages("ggplot2", repos="http://cran.us.r-project.org")'
#
# ========== Starts up script ==========
library(ggplot2)

args = commandArgs(TRUE)
seq_input1  = ifelse(length(args)>=1, args[1], "sequential_predictions_0_to_2.txt")
seq_input2  = ifelse(length(args)>=2, args[2], "sequential_predictions_1_to_2.txt")
seq_input3  = ifelse(length(args)>=3, args[3], "sequential_predictions_2_to_2.txt")

seq_output1 = ifelse(length(args)>=4, args[4], "sequential_violin_isic.pdf")

seq_input4  = ifelse(length(args)>=5, args[5], "sequential_predictions_0_to_4.txt")
seq_input5  = ifelse(length(args)>=6, args[6], "sequential_predictions_3_to_4.txt")
seq_input6  = ifelse(length(args)>=7, args[7], "sequential_predictions_4_to_4.txt")

seq_output2 = ifelse(length(args)>=8, args[8], "sequential_violin_edra.pdf")

# ========== Plots violin plot for isic ==========

# Fetches all results
seq1 = read.table(seq_input1, header=TRUE, sep=",")
seq2 = read.table(seq_input2, header=TRUE, sep=",")
seq3 = read.table(seq_input3, header=TRUE, sep=",")

drops = c("check_dataset", "sequence", "kickoff")
seq1 = seq1[, !(names(seq1) %in% drops)]
seq2 = seq2[, !(names(seq2) %in% drops)]
seq3 = seq3[, !(names(seq3) %in% drops)]
print(mean(seq1$isbi_auc))
print(mean(seq2$isbi_auc))
print(mean(seq3$isbi_auc))

seq = rbind(seq1,seq2,seq3)

seq$auc = with(seq, isbi_auc*100)
seq$sort_dataset <- factor(seq$sort_datase, levels=c("2", "1", "0"), labels=c("isic/test", "isic/val", "internal test split"))
colnames(seq) = c("hyperoptimization dataset", "auc", "auc (%)")

# pdf(seq_output1, width=7, height=4) # compact final version
#  ylim(70, 100) + # compact final version
pdf(seq_output1, width=7, height=7)
ggplot(data=seq,
    aes(x=`hyperoptimization dataset`, y=`auc (%)`)) +
  ylim(50, 100) +
  geom_violin() + geom_point(alpha=0.25, size=2) +
  stat_summary(fun.y=mean, geom="point", shape=20, size=10, color="red", fill="red", alpha=0.75) +
  theme_gray(base_size = 20)
dev.off()

# ========== Plots violin plot for edra ==========

# Fetches all results
seq1 = read.table(seq_input4, header=TRUE, sep=",")
seq2 = read.table(seq_input5, header=TRUE, sep=",")
seq3 = read.table(seq_input6, header=TRUE, sep=",")
print(mean(seq1$isbi_auc))
print(mean(seq2$isbi_auc))
print(mean(seq3$isbi_auc))

drops = c("check_dataset", "sequence", "kickoff")
seq1 = seq1[, !(names(seq1) %in% drops)]
seq2 = seq2[, !(names(seq2) %in% drops)]
seq3 = seq3[, !(names(seq3) %in% drops)]

seq = rbind(seq1,seq2,seq3)

seq$auc = with(seq, isbi_auc*100)
seq$sort_dataset <- factor(seq$sort_datase, levels=c("4", "3", "0"), labels=c("edra/clinic.", "edra/dermo.", "internal test split"))
colnames(seq) = c("hyperoptimization dataset", "auc", "auc (%)")

# pdf(seq_output2, width=7, height=4) # compact final version
#  ylim(50, 80) + # compact final version
pdf(seq_output2, width=7, height=7)
ggplot(data=seq,
    aes(x=`hyperoptimization dataset`, y=`auc (%)`)) +
  ylim(50, 100) +
  geom_violin() + geom_point(alpha=0.25, size=2) +
  stat_summary(fun.y=mean, geom="point", shape=20, size=10, color="red", fill="red", alpha=0.75) +
  theme_gray(base_size = 20)
dev.off()

# R -e 'install.packages("ggplot2", repos="http://cran.us.r-project.org")'
#
# ========== Starts up script ==========
library(ggplot2)
library(grid)
library(gtable)

args = commandArgs(TRUE)
meta_input1  = ifelse(length(args)>=1, args[1], "all_meta_predictions.txt")
meta_output1 = ifelse(length(args)>=2, args[2], "all_meta.pdf")
meta_input2  = ifelse(length(args)>=3, args[3], "zoomed_meta_predictions.txt")
meta_output2 = ifelse(length(args)>=4, args[4], "zoomed_meta.pdf")
meta_input3  = ifelse(length(args)>=5, args[5], "edra_meta_predictions.txt")
meta_output3 = ifelse(length(args)>=6, args[6], "edra_meta.pdf")
meta_input4  = ifelse(length(args)>=7, args[7], "isic_meta_predictions.txt")
meta_output4 = ifelse(length(args)>=8, args[8], "isic_meta.pdf")

classes_to_read=c("character","character","character","numeric","numeric","numeric","numeric")
drops = c("k_auc", "m_auc")

# ========== Plots complete series ==========

# Fetches all results
meta = read.table(meta_input1, header=TRUE, sep=",", colClasses=classes_to_read)

selmeta = meta
selmeta[meta$pool_method == "xtrm","pool_method"] = "extreme"
selmeta = selmeta[, !(names(selmeta) %in% drops)]
selmeta$auc = with(selmeta, isbi_auc*100)

selmeta$sort_dataset <- factor(selmeta$sort_dataset, levels=c("0", "random"), labels=c("internal test split", "random"))
selmeta$check_dataset <- factor(selmeta$check_dataset, levels=c("0", "1", "2", "3", "4"), labels=c("int. test split", "isic/val.", "isic/test", "edra/dermosc.", "edra/clinical"))

colnames(selmeta) = c("pooling: ", "sorted by", "tested on: ", "number of models", "auc", "auc (%)")

x=selmeta$`number of models`
y=selmeta$`auc (%)`

color_blind_proof=c(
  rgb(0,0,0, maxColorValue=255),
  rgb(0,73,73, maxColorValue=255),
  rgb(219,109,0, maxColorValue=255),
  rgb(146,0,0, maxColorValue=255),
  rgb(73,0,146, maxColorValue=255)
)

pdf(meta_output1, width=8, height=4)
ggplot(data=selmeta, aes(x=`number of models`, y=`auc (%)`, colour=`tested on: `, lty=`pooling: `)) +
  ylim(50, 100) +
  geom_line(size=1.25, alpha=0.8) + facet_wrap(~ `sorted by`) +
  guides(colour=guide_legend(direction="horizontal"), lty=guide_legend(direction="horizontal")) +
  theme_gray(base_size = 11) + theme(legend.position=c(0.42,0.175)) +
  scale_color_manual(values=color_blind_proof)
dev.off()


# ========== Plots edra series ==========

# Replicate color for 10 random series
all_colors=c(
  rgb(0,0,0, maxColorValue=255),
  rgb(0,73,73, maxColorValue=255),
  rgb(219,109,0, maxColorValue=255),
  rgb(146,0,0, maxColorValue=255),
  rgb(142,0,0, maxColorValue=255),
  rgb(143,0,0, maxColorValue=255),
  rgb(149,0,0, maxColorValue=255),
  rgb(141,0,0, maxColorValue=255),
  rgb(145,0,0, maxColorValue=255),
  rgb(150,0,0, maxColorValue=255),
  rgb(139,0,0, maxColorValue=255),
  rgb(151,0,0, maxColorValue=255),
  rgb(144,0,0, maxColorValue=255)
)

# Fetches all results
meta = read.table(meta_input3, header=TRUE, sep=",", colClasses=classes_to_read)

# ----- Just build legend -----

droprows = c('random2', 'random3', 'random4', 'random5', 'random6', 'random7', 'random8', 'random8', 'random9', 'random10')
selmeta = meta
selmeta = selmeta[!(selmeta$sort_dataset %in% droprows), !(names(selmeta) %in% drops)]
selmeta$auc = with(selmeta, isbi_auc*100)
selmeta$sort_dataset <- factor(selmeta$sort_dataset, levels=c("4", "3", "0", "random1"), labels=c("edra/clinical", "edra/dermosc.", "internal test split", "random"))
selmeta$check_dataset <- factor(selmeta$check_dataset, levels=c("4"), labels=c("edra/clin"))
colnames(selmeta) = c("pooling", "sorted by", "test", "number of models", "auc", "auc (%)")
(dummy <- ggplot(data=selmeta,
    aes(x=`number of models`, y=`auc (%)`, colour=`sorted by`)) +
  ylim(50, 100) +
  geom_line(size=2, alpha=0.8) + theme_gray(base_size = 20) + theme(legend.position=c(0.8,0.2)) +
  scale_color_manual(values=color_blind_proof))
color_legend <- gtable_filter(ggplot_gtable(ggplot_build(dummy)), "guide-box")

# ----------

selmeta = meta
selmeta = selmeta[, !(names(selmeta) %in% drops)]
selmeta$auc = with(selmeta, isbi_auc*100)

selmeta$sort_dataset <- factor(selmeta$sort_dataset,
    levels=c("4",             "3",             "0",                   "random1", "random2", "random3", "random4", "random5", "random6", "random7", "random8", "random9", "random10"),
    labels=c("edra/clinical", "edra/dermosc.", "internal test split", "random1", "random2", "random3", "random4", "random5", "random6", "random7", "random8", "random9", "random10"))

selmeta$check_dataset <- factor(selmeta$check_dataset, levels=c("4"), labels=c("edra/clin"))

colnames(selmeta) = c("pooling", "sorted by", "test", "number of models", "auc", "auc (%)")

# ggplot(data=selmeta,
#     aes(x=`number of models`, y=`auc (%)`, colour=`sorted by`)) +
#   ylim(50, 100) +
#   geom_line(size=2, alpha=0.8) + theme_gray(base_size = 20) + theme(legend.position=c(0.8,0.2)) +
#   scale_color_manual(values=color_blind_proof)
(plot <- ggplot(data=selmeta,
    aes(x=`number of models`, y=`auc (%)`, colour=`sorted by`)) +
  ylim(50, 100) +
  geom_line(size=2, alpha=0.8) + theme_gray(base_size = 20) + theme(legend.position="none") +
  scale_color_manual(values=all_colors))

pdf(meta_output3, width=7, height=7)
plot + annotation_custom(grob = color_legend, xmin = 4, xmax = 16, ymin = 50, ymax = 80)
dev.off()

# ========== Plots isic series ==========

# Fetches all results
meta = read.table(meta_input4, header=TRUE, sep=",", colClasses=classes_to_read)

# ----- Just build legend -----

droprows = c('random2', 'random3', 'random4', 'random5', 'random6', 'random7', 'random8', 'random8', 'random9', 'random10')
selmeta = meta
selmeta = selmeta[!(selmeta$sort_dataset %in% droprows), !(names(selmeta) %in% drops)]
selmeta$auc = with(selmeta, isbi_auc*100)
selmeta$sort_dataset <- factor(selmeta$sort_dataset, levels=c("2", "1", "0", "random1"), labels=c("isic/test", "isic/validation", "internal test split", "random"))
selmeta$check_dataset <- factor(selmeta$check_dataset, levels=c("4"), labels=c("edra/clin"))
colnames(selmeta) = c("pooling", "sorted by", "test", "number of models", "auc", "auc (%)")
(dummy <- ggplot(data=selmeta,
    aes(x=`number of models`, y=`auc (%)`, colour=`sorted by`)) +
  ylim(50, 100) +
  geom_line(size=2, alpha=0.8) + theme_gray(base_size = 20) + theme(legend.position=c(0.8,0.2)) +
  scale_color_manual(values=color_blind_proof))
color_legend <- gtable_filter(ggplot_gtable(ggplot_build(dummy)), "guide-box")

# ----------


selmeta = meta
selmeta = selmeta[, !(names(selmeta) %in% drops)]
selmeta$auc = with(selmeta, isbi_auc*100)

selmeta$sort_dataset <- factor(selmeta$sort_dataset,
    levels=c("2",         "1",               "0",                   "random1", "random2", "random3", "random4", "random5", "random6", "random7", "random8", "random9", "random10"),
    labels=c("isic/test", "isic/validation", "internal test split", "random1", "random2", "random3", "random4", "random5", "random6", "random7", "random8", "random9", "random10"))

selmeta$check_dataset <- factor(selmeta$check_dataset, levels=c("2"), labels=c("isic/test"))

colnames(selmeta) = c("pooling", "sorted by", "test", "number of models", "auc", "auc (%)")

# ggplot(data=selmeta,
#     aes(x=`number of models`, y=`auc (%)`, colour=`sorted by`)) +
#   ylim(50, 100) +
#   geom_line(size=2, alpha=0.8) + theme_gray(base_size = 20) + theme(legend.position=c(0.8,0.2)) +
#   scale_color_manual(values=all_colors)
(plot <- ggplot(data=selmeta,
    aes(x=`number of models`, y=`auc (%)`, colour=`sorted by`)) +
  ylim(50, 100) +
  geom_line(size=2, alpha=0.8) + theme_gray(base_size = 20) + theme(legend.position="none") +
  scale_color_manual(values=all_colors))

pdf(meta_output4, width=7, height=7)
plot + annotation_custom(grob = color_legend, xmin = 4, xmax = 16, ymin = 50, ymax = 80)
dev.off()




# R -e 'install.packages("corrgram", repos="http://cran.us.r-project.org")'
#
# ========== Starts up script ==========

library(corrgram)

args = commandArgs(TRUE)
jbhi_input      = ifelse(length(args)>=1, args[1], "all_results.anova.txt")
test_output1    = ifelse(length(args)>=2, args[2], "test_dataset_correlogram_isbi.pdf")
test_output2    = ifelse(length(args)>=3, args[3], "test_dataset_correlogram_melanoma.pdf")
metrics_output1 = ifelse(length(args)>=4, args[4], "metrics_correlogram_isic_test.pdf")
metrics_output2 = ifelse(length(args)>=5, args[5], "metrics_correlogram_edra_dermo.pdf")

# Fetches all results
jbhi = read.table(jbhi_input, header=TRUE, sep=";")

# ========== Customizes correlogram ==========

# All custompanel.* functions modified from original panel.* from corrgram.R

# ...customized scatterplot
custompanel.pts <- function(x, y, corr=NULL, col.regions, cor.method, ...){
  # For correlation matrix, do nothing
  if(!is.null(corr)) return()
  results = cor.test(x, y, conf.level=0.95, alternative = "two.sided")
  # First, the estimate
  est = results$estimate
  if (est>=0) {
    color = rgb(0,0,0,0.5)
    colorbox = "gray75"
  }
  else {
    color = rgb(1,0,0,0.5)
    colorbox = rgb(1,0.5,0.5)
  }
  plot.xy(xy.coords(x, y), type="p", pch=21, col="NA", bg=color)
  box(col=colorbox)
}

# ...customized correlation and confidence intervals
custompanel.conf <- function(x, y, corr=NULL, col.regions, cor.method, digits=0, cex.cor, ...){
  usr = par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))

  stopifnot(is.null(corr))
  # Calculate correlation and confidence interval
  if(sum(complete.cases(x,y)) < 4) {
    warning("Need at least 4 complete cases for cor.test()")
    return()
  }
  results = cor.test(x, y, conf.level=0.95, alternative = "two.sided")

  # First, the estimate
  est = results$estimate
  ci = results$conf.int

  plotSquare <- function(area, border="NA", fill=par("fg"), ...) {
    squareCoordinates <- function (area) {
      halfside = sqrt(area)/2.
      sidesX   = c(halfside, -halfside, -halfside,  halfside)
      sidesY   = c(halfside,  halfside, -halfside, -halfside)
      return(list(sidesX, sidesY))
    }
    centerX = c(0.5, 0.5, 0.5, 0.5)
    centerY = c(0.5, 0.5, 0.5, 0.5)
    sides = squareCoordinates(area)
    polygon(centerX+sides[[1]], centerY+sides[[2]], border=border, col=fill)
  }

  plotCircle <- function(area, maxarea=pi*0.5^2, border="NA", fill=par("fg"), lty=1, ...) {
    circleCoordinates <- function (radius, slices) {
      sequence = seq(0,slices)/slices
      sidesX = cos(sequence*2.*pi)*radius
      sidesY = sin(sequence*2.*pi)*radius
      return(list(sidesX, sidesY))
    }
    area = area*maxarea
    slices = 50
    centerX = rep(0.5, slices+1)
    centerY = rep(0.5, slices+1)
    radius = sqrt(area/pi)
    sides = circleCoordinates(radius, slices)
    polygon(centerX+sides[[1]], centerY+sides[[2]], border=border, col=fill, lty=lty)
    return(radius)
  }

  plotShape = plotCircle

  # text(0.5, 0.3, citext,  cex=cex.cor) # , col=pal[col.ind])
  # plot.new()
  # plotShape(ci[2], border="NA", fill="gray50", ...)
  # plotShape(est,   border="NA", fill="black", ...)
  # plotShape(ci[1], border="NA", fill="gray50", ...)
  if (est>=0) {
    color = "black"
    maxv = ci[2]
    midv = est
    minv = ci[1]
  }
  else {
    color = "red"
    maxv = -ci[1]
    midv = -est
    minv = -ci[2]
  }
  # print(paste("Maxv", maxv, "Midv", midv, "Minv", minv))
  midvt = formatC(midv*100., digits=digits, format='f')
  # size = 0.5/strwidth(midvt)
  if (minv>=0 && maxv>=0) {
    plotShape(maxv, border=color,   fill="NA",  lty=2, ...)
    radius = plotShape(midv, border=color,   fill=color, lty=1, ...)
    plotShape(minv, border="white", fill="NA",  lty=2, ...)
    size = radius/strwidth("99")
    text(0.5, 0.5, midvt, cex=size, col="white")
  }
  else {
    size = 0.5/strwidth("99")
    text(0.5, 0.5, midvt, cex=size, col=color)
  }
}

# ...customized text
custompanel.txt <- function(x=0.5, y=0.5, txt, cex, font, srt){
  text(x, y, txt, cex=cex, font=font, srt=srt)
}


# ...customized minimum and maximum
custompanel.minmax <- function(x, corr=NULL, addlabel="", ...){
  # For correlation matrix, do nothing
  if(!is.null(corr)) return()
  # Put the minimum in the lower-left corner and the
  # maximum in the upper-right corner
  minx  <- min(x, na.rm=TRUE)
  tminx <- paste(formatC(minx*100., digits=0, format='f'), " = ", addlabel, "min", sep="")
  maxx  <- max(x, na.rm=TRUE)
  tmaxx <- paste(addlabel, "max = ", formatC(maxx*100., digits=0, format='f'), sep="")
  text(minx, minx, tminx, cex=1, adj=c(0,0))
  text(maxx, maxx, tmaxx, cex=1, adj=c(1,1))
}

# ========== Plots correlogram of ISBI AUCs over dataset ==========

# Reformats the resutls table so that corresponding results
# on different datasets are side-by-side
drops = c("D4_N", "j")
jbhi0 = jbhi[jbhi$j==0, !(names(jbhi) %in% drops)]
jbhi1 = jbhi[jbhi$j==1, !(names(jbhi) %in% drops)]
jbhi2 = jbhi[jbhi$j==2, !(names(jbhi) %in% drops)]
jbhi3 = jbhi[jbhi$j==3, !(names(jbhi) %in% drops)]
jbhi4 = jbhi[jbhi$j==4, !(names(jbhi) %in% drops)]
colnames(jbhi0)[12:28] <- paste(colnames(jbhi0)[12:28], "0", sep="_")
colnames(jbhi1)[12:28] <- paste(colnames(jbhi1)[12:28], "1", sep="_")
colnames(jbhi2)[12:28] <- paste(colnames(jbhi2)[12:28], "2", sep="_")
colnames(jbhi3)[12:28] <- paste(colnames(jbhi3)[12:28], "3", sep="_")
colnames(jbhi4)[12:28] <- paste(colnames(jbhi4)[12:28], "4", sep="_")

mergecols = c("D1_N", "D3_N", "a", "b", "c", "d", "e", "f", "g", "h", "i")
jbhicorr = jbhi0
jbhicorr = merge(jbhicorr, jbhi1, by=mergecols)
jbhicorr = merge(jbhicorr, jbhi2, by=mergecols)
jbhicorr = merge(jbhicorr, jbhi3, by=mergecols)
jbhicorr = merge(jbhicorr, jbhi4, by=mergecols)

stopifnot(nrow(jbhicorr)==512)

# Selects and labels the columns for the correlogram
selcorr = jbhicorr[, c("isbi_auc_0", "isbi_auc_1", "isbi_auc_2", "isbi_auc_3", "isbi_auc_4")]
datasets = c("same-dataset\ntest split", "cross-dataset\nISIC/val", "cross-dataset\nISIC/test", "cross-dataset\nEDRA/dermo", "cross-dataset\nEDRA/clinic")

custompanel.minmaxauc <- function(...) {
  return(custompanel.minmax(addlabel="auc ", ...))
}

pdf(test_output1)
corrgram(selcorr, labels=datasets, order=FALSE,
    lower.panel=custompanel.conf,
    upper.panel=custompanel.pts,
    diag.panel =custompanel.minmaxauc,
    text.panel =custompanel.txt,
    cor.method ='spearman')
dev.off()

# ========== Plots correlogram of melanoma AUCs over dataset ==========

# Reformats the resutls table so that corresponding results
# on different datasets are side-by-side
drops = c("D4_N", "j")
jbhi0 = jbhi[jbhi$j==0, !(names(jbhi) %in% drops)]
jbhi1 = jbhi[jbhi$j==1, !(names(jbhi) %in% drops)]
jbhi2 = jbhi[jbhi$j==2, !(names(jbhi) %in% drops)]
jbhi3 = jbhi[jbhi$j==3, !(names(jbhi) %in% drops)]
jbhi4 = jbhi[jbhi$j==4, !(names(jbhi) %in% drops)]
colnames(jbhi0)[12:28] <- paste(colnames(jbhi0)[12:28], "0", sep="_")
colnames(jbhi1)[12:28] <- paste(colnames(jbhi1)[12:28], "1", sep="_")
colnames(jbhi2)[12:28] <- paste(colnames(jbhi2)[12:28], "2", sep="_")
colnames(jbhi3)[12:28] <- paste(colnames(jbhi3)[12:28], "3", sep="_")
colnames(jbhi4)[12:28] <- paste(colnames(jbhi4)[12:28], "4", sep="_")

mergecols = c("D1_N", "D3_N", "a", "b", "c", "d", "e", "f", "g", "h", "i")
jbhicorr = jbhi0
jbhicorr = merge(jbhicorr, jbhi1, by=mergecols)
jbhicorr = merge(jbhicorr, jbhi2, by=mergecols)
jbhicorr = merge(jbhicorr, jbhi3, by=mergecols)
jbhicorr = merge(jbhicorr, jbhi4, by=mergecols)

stopifnot(nrow(jbhicorr)==512)

# Selects and labels the columns for the correlogram
selcorr = jbhicorr[, c("m_auc_0", "m_auc_1", "m_auc_2", "m_auc_3", "m_auc_4")]
datasets = c("same-dataset\ntest split", "cross-dataset\nISIC/val", "cross-dataset\nISIC/test", "cross-dataset\nEDRA/dermo", "cross-dataset\nEDRA/clinic")

custompanel.minmaxauc <- function(...) {
  return(custompanel.minmax(addlabel="auc ", ...))
}

pdf(test_output2)
corrgram(selcorr, labels=datasets, order=FALSE,
    lower.panel=custompanel.conf,
    upper.panel=custompanel.pts,
    diag.panel =custompanel.minmaxauc,
    text.panel =custompanel.txt,
    cor.method ='spearman')
dev.off()

# ========== Plots correlogram of metrics on ISIC Test Split ==========

# Selects and labels the columns for the correlogram
selcorr = jbhi[jbhi$j==2,]
selcorr$m_tnr = with(selcorr, 1.-m_fpr)
selcorr$k_tnr = with(selcorr, 1.-k_fpr)
selcorr = selcorr[, c("m_ap", "m_auc", "m_tpr", "m_tnr", "k_ap", "k_auc", "k_tpr", "k_tnr")]
metrics = c("melanoma\navg. prec.",  "melanoma\nauc" ,  "melanoma\nsensitivity",  "melanoma\nspecificity",
            "keratosis\navg. prec.", "keratosis\nauc" , "keratosis\nsensitivity", "keratosis\nspecificity")

#stopifnot(nrow(selcorr)==2560)
stopifnot(nrow(selcorr)==512)

pdf(metrics_output1)
corrgram(selcorr, labels=metrics, order=FALSE,
    lower.panel=custompanel.conf,
    upper.panel=custompanel.pts,
    diag.panel =custompanel.minmax,
    text.panel =custompanel.txt,
    cor.method ='spearman')
dev.off()

# ========== Plots correlogram of metrics on EDRA / dermoscopic images ==========

# Selects and labels the columns for the correlogram
selcorr = jbhi[jbhi$j==3,]
selcorr$m_tnr = with(selcorr, 1.-m_fpr)
selcorr$k_tnr = with(selcorr, 1.-k_fpr)
selcorr = selcorr[, c("m_ap", "m_auc", "m_tpr", "m_tnr", "k_ap", "k_auc", "k_tpr", "k_tnr")]
metrics = c("melanoma\navg. prec.",  "melanoma\nauc" ,  "melanoma\nsensitivity",  "melanoma\nspecificity",
            "keratosis\navg. prec.", "keratosis\nauc" , "keratosis\nsensitivity", "keratosis\nspecificity")

#stopifnot(nrow(selcorr)==2560)
stopifnot(nrow(selcorr)==512)

pdf(metrics_output2)
corrgram(selcorr, labels=metrics, order=FALSE,
    lower.panel=custompanel.conf,
    upper.panel=custompanel.pts,
    diag.panel =custompanel.minmax,
    text.panel =custompanel.txt,
    cor.method ='spearman')
dev.off()

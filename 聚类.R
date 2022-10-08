library(survival)
library(survminer)
library(dplyr)
library(glmnet)
library(NbClust)
library(circlize)
library(survival)
library(survminer)
library(ggthemes)
library(tidyverse)
library(ggpubr)
library(RColorBrewer)
library(ConsensusClusterPlus)

windowsFonts(arial=windowsFont("arial"))
setwd('C:/Users/rog/Desktop/聚类分析')
TNBC_M_count <- read.table(file = 'C:/Users/rog/Desktop/取集合/system_metabric_data.txt', header = T, check.names = F)
rownames(TNBC_M_count) <- TNBC_M_count[, 1]
TNBC_M_count <- TNBC_M_count[, -1]
TNBC_M_clinical <- read.csv(file = 'C:/Users/rog/Desktop/预后/Metabric_TNBC_clinical.csv', header = T, sep = '\t', check.names = F)

TNBC_metabric_clinical_2 <- TNBC_M_clinical[, c('Patient ID', 'Overall Survival (Months)', "Patient's Vital Status")]
TNBC_M_count_1 <- t(TNBC_M_count)
TNBC_M_count_1 <-as.data.frame(TNBC_M_count_1)
TNBC_M_count_1$`Patient ID` <- rownames(TNBC_M_count_1)
TNBC_M_count_1 <- TNBC_M_count_1[,c(27, 1: 26)]
metabric_counts_clinical <- merge(TNBC_metabric_clinical_2, TNBC_M_count_1
                                  , by.x = 'Patient ID'
                                  , by.y = 'Patient ID'
)
TNBC_metabric_count_clinical <- metabric_counts_clinical[(metabric_counts_clinical$`Patient's Vital Status` == 'Living'), ]
TNBC_metabric_count_clinical_1 <- metabric_counts_clinical[(metabric_counts_clinical$`Patient's Vital Status` == 'Died of Disease'), ]
TNBC_metabric_count_clinical_2 <- rbind(TNBC_metabric_count_clinical, TNBC_metabric_count_clinical_1)
TNBC_metabric_count_clinical_2[TNBC_metabric_count_clinical_2$`Patient's Vital Status` == 'Died of Disease',]$`Patient's Vital Status` <- 1
TNBC_metabric_count_clinical_2[TNBC_metabric_count_clinical_2$`Patient's Vital Status` == 'Living',]$`Patient's Vital Status` <- 0
TNBC_metabric_count_clinical_2$`Overall Survival (Months)` <- as.numeric(TNBC_metabric_count_clinical_2$`Overall Survival (Months)`)
TNBC_metabric_count_clinical_2$`Patient's Vital Status` <- as.numeric(TNBC_metabric_count_clinical_2$`Patient's Vital Status`)
rownames(TNBC_metabric_count_clinical_2) <- TNBC_metabric_count_clinical_2$`Patient ID`
TNBC_metabric_count_clinical_2 <- TNBC_metabric_count_clinical_2[, -1]
names(TNBC_metabric_count_clinical_2)[names(TNBC_metabric_count_clinical_2) == 'Overall Survival (Months)'] <- 'futime'
names(TNBC_metabric_count_clinical_2)[names(TNBC_metabric_count_clinical_2) == "Patient's Vital Status"] <- 'fustat'
rt <- TNBC_metabric_count_clinical_2

rt <- rt[, c(8, 12, 16, 20, 25)]

#这里采用中位数绝对差mad值，进行中位数中心化构建模型
rt <- as.data.frame(t(rt))
mads=apply(rt,1,mad)
rt = sweep(rt,1, apply(rt,1,median,na.rm=T))
rt =as.matrix(rt)
gene = rownames(rt)
title=("JULEI")
set.seed(66)
results = ConsensusClusterPlus(rt,maxK=9,
                               reps=1000,
                               pItem=0.8,
                               pFeature=1,
                               title=title,
                               clusterAlg="km",
                               distance="euclidean",seed=1,
                               plot="pdf",
                               tmyPal = c('white', '#1E7DC1')
                               )
icl = calcICL(results,title=title,plot="pdf")
#查看聚类后分组的信息
group<-results[[2]][["consensusClass"]]
group<-as.data.frame(group)
group$group <- factor(group$group,levels=c(1,2))#聚类成多少组则写多少组
save(group,file = "group.Rdata")
write.table(group,file = "ClassGroup.txt",sep = "\t",row.names = T,
            col.names = NA,quote = F)



data=TNBC_metabric_count_clinical_2
data <- data[, c(1, 2, 8, 12, 16, 20, 25)]
data <- cbind(data, group)
data$group=factor(data$group, levels=c("1", "2"))


data <- read.table(file = "class_data.txt",sep = "\t", header = T, row.names = 1)
cox <- coxph(Surv(futime, fustat) ~ data$group, data = data)
coxSummary = summary(cox)
HR=coxSummary$conf.int[,"exp(coef)"]
HR.95L=coxSummary$conf.int[,"lower .95"]
HR.95H=coxSummary$conf.int[,"upper .95"]
pvalue=coxSummary$coefficients[,"Pr(>|z|)"]
data[data$group == 1, ]$group <- 'MCA'
data[data$group == 2, ]$group <- 'MCB'
data$group <- as.factor(data$group)


diff=survdiff(Surv(futime, fustat) ~ group, data = data)
length=length(levels(factor(data[,"group"])))
pValue=1-pchisq(diff$chisq, df=length-1)

fit <- survfit(Surv(futime, fustat) ~ group, data = data)
bioCol=c('#1E7DC1','#CC4D65',"#6E568C","#7CC767","#223D6C","#D20A13","#FFD121","#088247","#11AA4D")
bioCol=bioCol[1:length]

p=ggsurvplot(fit, 
             data=data,
             conf.int=F,
             pval=TRUE,
             pval.size=5,
             legend.labs=levels(factor(data[,"group"])),
             legend = c(0.4, 0.27),
             font.legend=14,
             xlab="Time (Months)",
             palette = bioCol,
             pval.method = TRUE, 
             cumevents=F
             , break.time.by = 50
             , ggtheme = theme_few()
             ,font.tickslab = c(16, "black")
)
p[["plot"]][["theme"]][["axis.title.x"]][["family"]] = 'arial'  #x轴标题
p[["plot"]][["layers"]][[4]][["computed_geom_params"]][["family"]] = 'arial'
p[["plot"]][["layers"]][[5]][["aes_params"]][["size"]] = 5
p[["plot"]][["layers"]][[4]][["computed_geom_params"]][["family"]] = 'italic'
p[["plot"]][["theme"]][["axis.title.y"]][["family"]] = 'arial'
p[["plot"]][["layers"]][[4]][["aes_params"]][["label"]] <- expression(italic('P = 0.0239'))
p[["plot"]][["theme"]][["axis.text.x"]][["size"]] = 14
p[["plot"]][["theme"]][["axis.text.y"]][["size"]] = 14
p[["table"]][["layers"]][[1]][["aes_params"]][['family']] <- 'sans'
p[["table"]][["theme"]][["axis.text"]][["colour"]] <- 'black'
p[["plot"]][["theme"]][["panel.background"]][["fill"]] <- NA
p[["plot"]][["theme"]][["axis.title.y"]][["size"]] <- 16
p[["plot"]][["theme"]][["axis.title.x"]][["size"]] <- 16
p[["plot"]][["theme"]][["axis.ticks"]][["size"]] <- 0.5
p
ggsave(filename = 'C:/Users/rog/Desktop/聚类分析/聚类组生存曲线.pdf',width = 5,height = 4.5)

write.table(data,file = "class_data.txt",sep = "\t",row.names = T,
            col.names = NA,quote = F)




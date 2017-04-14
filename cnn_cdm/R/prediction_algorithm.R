library("randomForest")
library("RODBC")
library(dplyr)
library("DMwR")
library("pROC")
library(CohortMethod) 
library(SqlRender)

############################################################################################
#####Re-try#################################################################################
############################################################################################
############################################################################################
rm(list=ls())
gc()

getwd()
setwd("C:/Users/ABMI/OneDrive/Study/DL_CVDPrediction/result")

connectionDetails<-createConnectionDetails(dbms="sql server",
                                           server="128.1.99.53",
                                           schema="CHAN_NHID_CVD.dbo",
                                           user="chandryou",
                                           password="dbtmdcks12#")
#set databaseSchema,train_cohort_schema, train_dx schema
databaseSchema<-"CHAN_NHID_CVD.dbo"
whole_cohort<-"cohort"
train_cohort<-"train_cohort"
test_cohort<-"test_cohort"

dx_cohort<-"temp_pre_condition"
drug_cohort<-"drug_cohort"

train_dx<-"pre_condition_train"
test_dx<-"pre_condition_test"

train_drug<-"drug_train"
test_drug<-"drug_test"

workFolder <- getwd()

connection<-connect(connectionDetails)

##get whole cohort
sql<-("SELECT * FROM @databaseSchema.@whole_cohort")
sql<-(renderSql(sql,
                databaseSchema=databaseSchema,
                whole_cohort=whole_cohort)$sql)
sql<-translateSql(sql,
                  targetDialect=connectionDetails$dbms)$sql
cohort<-querySql(connection, sql)

cohort$COHORT_START_DATE<-NULL
cohort$COHORT_END_DATE<-NULL
cohort$CARDIO_DEATH<-NULL
cohort$MI<-NULL
cohort$CVA<-NULL


##get diagnosis cohort person_id
sql<-("SELECT DISTINCT PERSON_ID FROM @databaseSchema.@dx_cohort")
sql<-(renderSql(sql,
                databaseSchema=databaseSchema,
                dx_cohort=dx_cohort)$sql)
sql<-translateSql(sql,
                  targetDialect=connectionDetails$dbms)$sql
dx_cohort_person<-querySql(connection, sql)
dx_cohort_person<-unique(dx_cohort_person[[1]])

##get test cohort person_id
sql<-("SELECT DISTINCT PERSON_ID FROM @databaseSchema.@drug_cohort")
sql<-(renderSql(sql,
                databaseSchema=databaseSchema,
                drug_cohort=drug_cohort)$sql)
sql<-translateSql(sql,
                  targetDialect=connectionDetails$dbms)$sql
drug_cohort_person<-querySql(connection, sql)
drug_cohort_person<-unique(drug_cohort_person[[1]])

#remove patients without previous medication or drug history

# nrow(cohort) = 501703
cohort<-cohort %>%
        filter(PERSON_ID %in% dx_cohort_person & PERSON_ID %in% drug_cohort_person)

#nrow(cohort) = 442898
rm(drug_cohort_person)
rm(dx_cohort_person)

###########################################################
##Diagnosis#####################################################
###########################################################
##making diagnosis master table and unique diagnosis set

sql<-("SELECT distinct condition_concept_id, kcd_class FROM @databaseSchema.@dx_cohort")
sql<-(renderSql(sql,
                databaseSchema=databaseSchema,
                dx_cohort=dx_cohort)$sql)
sql<-translateSql(sql,
                  targetDialect=connectionDetails$dbms)$sql
master_dx<-querySql(connection, sql)

#delete NULL data
master_dx<-master_dx[!is.na(master_dx$CONDITION_CONCEPT_ID),]
dx_class<-unique(master_dx$KCD_CLASS)

##length(dx_class) : 159 (disease class)
##nrow(master_dx) :3687 (whole disease)

total_start_time<-Sys.time()
dx_cohort_featured<-cohort
for (i in 1:length(dx_class)){
        start_time<-Sys.time()
        #set disease class
        oneDx <- dx_class [i]
        classified_dx<-master_dx %>%
                filter(KCD_CLASS == oneDx) %>%
                select(CONDITION_CONCEPT_ID)
        classified_dx<-classified_dx[[1]]
        
        #Get sum of number of patients with the disease (if less than 1000 patients->next!)
        sql<-(paste ("SELECT COUNT(distinct person_id) FROM @databaseSchema.@dx_cohort where kcd_class ='",oneDx, "'",sep=""))
        sql<-(renderSql(sql,
                        databaseSchema=databaseSchema,
                        dx_cohort=dx_cohort)$sql)
        sql<-translateSql(sql,
                          targetDialect=connectionDetails$dbms)$sql
        
        count_dx_class<-querySql(connection, sql)
        
        if (count_dx_class< 50000)  next
        cohort_featured<-data.frame(PERSON_ID=cohort[,1])
        for (j in 1:length(classified_dx)){
                #get cohort with the specific disease in the class
                
                sql<-(paste ("SELECT person_id,total_dx FROM @databaseSchema.@dx_cohort where condition_concept_id =",classified_dx[j], sep=""))
                
                sql<-(renderSql(sql,
                                databaseSchema=databaseSchema,
                                dx_cohort=dx_cohort)$sql)
                sql<-translateSql(sql,
                                  targetDialect=connectionDetails$dbms)$sql
                dx_cohort_set<-querySql(connection, sql)
                if(unique(dx_cohort_set$PERSON_ID < 5000 )) next
                
                cohort_featured<-merge (x=cohort_featured,y=dx_cohort_set, by="PERSON_ID",all.x=TRUE, all.y=FALSE)
                cohort_featured[is.na(cohort_featured [,ncol(cohort_featured)] ),ncol(cohort_featured)]<-0
                colnames(cohort_featured)[length(cohort_featured)]<-paste("Dx",classified_dx[j],sep="_")
        }
        res <- prcomp(cohort_featured[,-1], center = TRUE, scale = FALSE)
        
        pc.use<-sum((cumsum(res$sdev^2/sum(res$sdev^2))>0.8)==FALSE)
        if(pc.use==0) pc.use<-pc.use+1
        
        tt<-predict(res)[, 1:pc.use]
        tt2<-scale(cohort_featured[,-1],res$center, res$scale ) %*% res$rotation[, 1:pc.use]
        
        cohort_featured<-cbind("PERSON_ID"=cohort_featured[,1],tt2)
        
        
        colnames(cohort_featured)[ (ncol(cohort_featured)-pc.use+1) : ncol(cohort_featured)] <- paste(rep(paste("Dx",oneDx,sep="_"),pc.use),1:pc.use, sep="_")
        dx_cohort_featured<-merge(x=cohort_featured, y=dx_cohort_featured, by="PERSON_ID",all.x=TRUE,all.y=FALSE )
        end_time<-Sys.time()
        elapsedtime<-end_time-start_time
        size_MB<-round(object.size(dx_cohort_featured)/(1024*1024),digits=1)
        print(paste(i,"th Diagnosis",oneDx,"was done during", round(elapsedtime,digits=1),attributes(elapsedtime)$units,"/ size =",size_MB,"MB" ))
        gc()
}


total_end_time<-Sys.time()
total_elapsed_time_dx<-total_end_time-total_start_time


saveRDS(dx_cohort_featured, file= "dx_cohort_featured.rds")

############################################################
###########################################################
rm(list=ls())
gc()
###########################################################
##DRUG#####################################################
###########################################################

getwd()
setwd("C:/Users/ABMI/OneDrive/Study/DL_CVDPrediction/result")

connectionDetails<-createConnectionDetails(dbms="sql server",
                                           server="128.1.99.53",
                                           schema="CHAN_NHID_CVD.dbo",
                                           user="chandryou",
                                           password="dbtmdcks12#")
#set databaseSchema,train_cohort_schema, train_dx schema
databaseSchema<-"CHAN_NHID_CVD.dbo"
whole_cohort<-"cohort"
train_cohort<-"train_cohort"
test_cohort<-"test_cohort"

dx_cohort<-"temp_pre_condition"
drug_cohort<-"drug_cohort"

train_dx<-"pre_condition_train"
test_dx<-"pre_condition_test"

train_drug<-"drug_train"
test_drug<-"drug_test"

workFolder <- getwd()

connection<-connect(connectionDetails)

##get whole cohort
sql<-("SELECT * FROM @databaseSchema.@whole_cohort")
sql<-(renderSql(sql,
                databaseSchema=databaseSchema,
                whole_cohort=whole_cohort)$sql)
sql<-translateSql(sql,
                  targetDialect=connectionDetails$dbms)$sql
cohort<-querySql(connection, sql)

cohort$COHORT_START_DATE<-NULL
cohort$COHORT_END_DATE<-NULL
cohort$CARDIO_DEATH<-NULL
cohort$MI<-NULL
cohort$CVA<-NULL


##making drug master table and drug class list
sql<-("SELECT distinct DRUG_CLASS,CONCEPT_ID FROM @databaseSchema.drug_master")
sql<-(renderSql(sql,
                databaseSchema=databaseSchema,
                train_dx=train_dx)$sql)
sql<-translateSql(sql,
                  targetDialect=connectionDetails$dbms)$sql
drug_master<-querySql(connection, sql)

#delete NA value ( no NA values in this study)
drug_master<-drug_master[!is.na(drug_master$CONCEPT_ID),]
nrow(drug_master)
#make unique drug_class list
drug_class<-unique(drug_master$DRUG_CLASS)
#length(drug_class)  : 92 (drug class)

total_start_time<-Sys.time()
drug_cohort_featured<-cohort

#i<-7
#j<-7

for (i in 1:length(drug_class)){
        start_time<-Sys.time()
        
        #set drug class
        oneDrug <- drug_class [i]
        classified_drug<-drug_master %>%
                filter(DRUG_CLASS == oneDrug) %>%
                select(CONCEPT_ID)
        classified_drug<-classified_drug[[1]]
        
        #Get sum of number of patients with the disease (if less than 1000 patients->next!)
        sql<-(paste ("SELECT COUNT(distinct person_id) FROM @databaseSchema.@drug_cohort where drug_concept_id in (",paste(classified_drug, collapse=","), ")",sep=""))
        sql<-(renderSql(sql,
                        databaseSchema=databaseSchema,
                        drug_cohort=drug_cohort)$sql)
        sql<-translateSql(sql,
                          targetDialect=connectionDetails$dbms)$sql
        count_drug_class<-querySql(connection, sql)
        
        if (count_drug_class< 10000) next
        cohort_featured<-data.frame(PERSON_ID=cohort[,1])
        for (j in 1:length(classified_drug)){
                
                #get whole cohort with the specific drug in the class (including train and test)
                
                sql<-(paste ("SELECT * FROM @databaseSchema.@drug_cohort where drug_concept_id =",classified_drug[j], sep=""))
                sql<-(renderSql(sql,
                                databaseSchema=databaseSchema,
                                drug_cohort=drug_cohort)$sql)
                sql<-translateSql(sql,
                                  targetDialect=connectionDetails$dbms)$sql
                drug_x<-querySql(connection, sql)
                
                if ( length(drug_x$PERSON_ID)< 500 ) next
                
                drug_x[,c(4:10)]<-drug_x[,c(4:10)]/drug_x$TOTAL_SUM
                
                clusters<-kmeans(drug_x[,c(4:10)],9, iter.max=30)
                if(clusters$ifault==4){
                        clusters<-kmeans(drug_x[,c(4:10)],9,algorithm = "MacQueen")
                }
                drug_x$cluster<-clusters$cluster
                
                drug_x<-drug_x %>%
                        select(PERSON_ID, TOTAL_SUM,cluster)
                
                cohort_featured<-merge (x=cohort_featured,y=drug_x, by="PERSON_ID",all.x=TRUE, all.y=FALSE)
                
                cohort_featured[is.na(cohort_featured [,ncol(cohort_featured)-1] ),ncol(cohort_featured)-1]<-0
                cohort_featured[is.na(cohort_featured [,ncol(cohort_featured)] ),ncol(cohort_featured)]<-0
                
                colnames(cohort_featured)[length(cohort_featured)-1]<-paste("sum",classified_drug[j],sep="_")
                colnames(cohort_featured)[length(cohort_featured)]<-paste("cluster",classified_drug[j],sep="_")
                
        }
        
        res <- prcomp(cohort_featured[,-1], center = TRUE, scale = FALSE)
        plot(cumsum(res$sdev^2/sum(res$sdev^2)))
        pc.use<-sum((cumsum(res$sdev^2/sum(res$sdev^2))>0.8)==FALSE)
        if(pc.use==0) pc.use<-pc.use+1
        
        tt<-predict(res)[, 1:pc.use]
        tt2<-scale(cohort_featured[,-1],res$center, res$scale ) %*% res$rotation[, 1:pc.use]
        
        cohort_featured<-cbind("PERSON_ID"=cohort_featured[,1],tt2)
        
        colnames(cohort_featured)[ (ncol(cohort_featured)-pc.use+1) : ncol(cohort_featured)] <- paste(rep(paste("drug",oneDrug,sep="_"),pc.use),1:pc.use, sep="_")
        drug_cohort_featured<-merge(x=cohort_featured, y=drug_cohort_featured, by="PERSON_ID",all.x=TRUE,all.y=FALSE )
        
        end_time<-Sys.time()
        elapsedtime<-end_time-start_time
        size_MB<-object.size(drug_cohort_featured)/(1024*1024)
        print(paste(i,"th drug was done", elapsedtime,attributes(elapsedtime)$units,"/ size =",size_MB,"MB" ))
        gc()
        
}
saveRDS(drug_cohort_featured, file= "drug_cohort_featured.rds")

##see plotting how drug was clustered####################################
#temp<-drug_x[1,]
#plot(x=c(2:8), y=as.vector(temp[,4:10]),type="l",col=temp$cluster,ylim=c(0,1.0))
#for(i in 2:100){
#        temp<-drug_x[i,]
#        lines(x=c(2:8), y=as.vector(temp[,4:10]),type="l", col=temp$cluster)
#}
########################################################################################
##########################################################################################

dx_cohort_featured<-readRDS(file= "dx_cohort_featured.rds")
dx_cohort_featured<-dx_cohort_featured[,1: (ncol(dx_cohort_featured)-3)]

drug_cohort_featured<-readRDS(file= "drug_cohort_featured.rds")

cohort_featured<-merge(x=dx_cohort_featured, y=drug_cohort_featured, by="PERSON_ID",all.x=TRUE,all.y=FALSE )

saveRDS(cohort_featured, file= "cohort_featured.rds")

ncol(dx_cohort_featured) # 198 (diagnosis feature : 197)
ncol(drug_cohort_featured) #160 (drug feature : 156)

##get train cohort person_id
sql<-("SELECT DISTINCT PERSON_ID FROM @databaseSchema.@train_cohort")
sql<-(renderSql(sql,
                databaseSchema=databaseSchema,
                train_cohort=train_cohort)$sql)
sql<-translateSql(sql,
                  targetDialect=connectionDetails$dbms)$sql
train_cohort_person<-querySql(connection, sql)
train_cohort_person<-train_cohort_person[[1]]

##get test cohort person_id
sql<-("SELECT DISTINCT PERSON_ID FROM @databaseSchema.@test_cohort")
sql<-(renderSql(sql,
                databaseSchema=databaseSchema,
                test_cohort=test_cohort)$sql)
sql<-translateSql(sql,
                  targetDialect=connectionDetails$dbms)$sql
test_cohort_person<-querySql(connection, sql)
test_cohort_person<-test_cohort_person[[1]]


cohort_featured_train<-cohort_featured %>%
        filter(PERSON_ID%in%train_cohort_person)

cohort_featured_test<-cohort_featured %>%
        filter(PERSON_ID%in%test_cohort_person)

#nrow(cohort_featured)#455543
#nrow(cohort_featured_train) #410022

#nrow(cohort_featured_test) #45521
#sum(cohort_featured_train$MACCE)
#sum(cohort_featured_test$MACCE)

#5 fold cross validation
sep_point<- cut(seq(1,nrow(cohort_featured_train)),breaks=5,labels=FALSE)
validation_set<-cohort_featured_train[sep_point==1,]
train_set<-cohort_featured_train[sep_point!=1,]


#(oversampling of pt with MACCE, undersampling of pt without MACCE )
trn_1<-train_set%>%
        filter(MACCE==0)%>%
        sample_frac (0.1)
trn_2<- train_set%>%
        filter(MACCE==1)
train_set2<-rbind(trn_1,trn_2,trn_2,trn_2,trn_2,trn_2,trn_2)


train_set2$MACCE<-as.factor(train_set2$MACCE)

fit<-randomForest(MACCE~.,data=train_set2[,-1],ntree=200)
plot(fit)
fit

predicted = cbind(predict(fit,validation_set,type="prob"),"PERSON_ID"=validation_set$PERSON_ID,"MACCE"=validation_set$MACCE)
predicted<-data.frame(predicted)
auc<-roc(predicted$MACCE,predicted[,1])
plot.roc(auc)
auc

predicted = cbind(predict(fit,cohort_featured_test,type="prob"),"PERSON_ID"=cohort_featured_test$PERSON_ID,"MACCE"=cohort_featured_test$MACCE)
predicted<-data.frame(predicted)
auc<-roc(predicted$MACCE,predicted[,1])
plot.roc(auc)
auc

saveRDS(auc, file= "final_auc.rds")

ro<-c(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
for (i in 1:9){
        cutoff<-ro[i]
        
        TP<- sum(predicted$MACCE==1 & predicted[,2]>=cutoff ) 
        TN<- sum(predicted$MACCE==0 & predicted[,2]< cutoff ) 
        FP<- sum(predicted$MACCE==0 & predicted[,2]>= cutoff ) 
        FN<- sum(predicted$MACCE==1 & predicted[,2]< cutoff ) 
        PPV<-round(TP/(TP+FP),digits=2)
        NPV<-round(TN/(TN+FN),digits=2)
        sensitivity <- round(TP/(TP+FN),digits =2 )
        specificity <- round(TN/(FP+TN), digits = 2)
        target_population<-round( (TP+FP)*100 / nrow(predicted))
        print(paste("cutoff=",cutoff,", target population=",target_population,"% of total / " ,
                    "PPV = ",PPV," NPV = ",NPV," sensitivity = ",sensitivity," specificity = ",specificity, sep=""))
}
##Until here



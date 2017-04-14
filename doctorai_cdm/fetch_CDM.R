library(CohortMethod) 
library(SqlRender)
######################################################
##YOU NEED TO FILL !!#################################

#set working folder as git
workFolder<-"../ajou_deep/doctorai_cdm"
setwd(workFolder)

connectionDetails<-createConnectionDetails(dbms="sql server",
                                           server=" ",
                                           schema="",
                                           user="",
                                           password="  ")
cdmVersion <- "5" 
cdmDatabaseSchema <- ""
resultsDatabaseSchema <- ""
cohort = "cohort"


#######################################################
#######################################################

###connection##########################################
connection<-connect(connectionDetails)

####CREATE TABLE####################################################
sql <- readSql( paste(workFolder,"/sql/cohort_extraction.sql",sep="") )
sql <- renderSql(sql,
                 cdmDatabaseSchema=cdmDatabaseSchema,
                 resultsDatabaseSchema=resultsDatabaseSchema,
                 cohort=cohort,
                 cohort_definition_id = 40,
                 outcome_id= 1)$sql
sql <- translateSql(sql,
                    targetDialect=connectionDetails$dbms)$sql
cohort_tab<-querySql(connection, sql)
#################################################################

##change column names into lower case
colnames(cohort_tab)<-tolower(colnames(cohort_tab))

##remove na values in table
cohort_tab<-(na.omit(cohort_tab))

#make one-hot vector matrix (df)
cohort_tab$condition_concept_id<-as.factor(cohort_tab$condition_concept_id)
df<-data.frame(with(cohort_tab, model.matrix(~ condition_concept_id + 0)))
colnames(df)<-paste("dx_",seq(length(df)),sep="")

#merge original table and one-hot encoding table
encoded<-aggregate(df, by = list(cohort_tab$person_id, cohort_tab$outcome, cohort_tab$gender_concept_id, cohort_tab$year_of_birth, cohort_tab$visit_seq, cohort_tab$visit_start_date),FUN=sum )
colnames(encoded)[1:6]<-c("person_id", "outcome", "gender", "age" ,"visit_seq", "visit_start_date")

#age and gender -> regularization
encoded$age<-as.numeric(format(encoded$visit_start_date, '%Y'))-encoded$age
encoded$age <- (encoded$age - mean(encoded$age)) / sd(encoded$age)
    #hist(encoded$age)
    #gender concept_id 8532 : female / 8507 : male
encoded$gender<-ifelse(encoded$age == 8532, 1, 0)

#ceiling one-hot vector 1 or under
encoded[7:ncol(encoded)] <- sapply(encoded[7:ncol(encoded)], function(x) { as.numeric(x > 0) })

#adding date difference between visit(day)
encoded = encoded[order(encoded$person_id, encoded$visit_seq),]
encoded$datediff = (encoded$visit_seq>1) * as.integer(encoded$visit_start_date - append(0, encoded$visit_start_date)[1:length(encoded$visit_start_date)])
encoded<-encoded[, c(1:5, ncol(encoded), 6: (ncol(encoded)-1))]

encoded$person_id<-as.numeric(as.factor(encoded$person_id))

#save RDS
saveRDS(encoded,paste(workFolder,"/data/encodedcohort.rds",sep=""))

##readRDS##
#encoded<-readRDS(paste(workFolder,"/data/encodedcohort.rds",sep=""))

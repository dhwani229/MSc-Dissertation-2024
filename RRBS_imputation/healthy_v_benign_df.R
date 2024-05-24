# making the dataframe of all the coverage files in R for healthy vs benign samples

library(methylKit)

file_path <- "/data/scratch/bt22912/prac/healthyvbenign.csv"
data <- read.csv(file_path)


sample.id <- as.list(data$Sample)
treatment <- as.factor(data$Treatment)
file.list <- as.list(data$Path)


myobj <- methRead(file.list,
                  sample.id = sample.id,
                  treatment = treatment,
                  assembly = "hg38",
                  context = "CpG",
                  pipeline = "bismarkCoverage",
                  mincov = 3)


print(myobj)
filtered_myobj <- filterByCoverage(myobj,
                                   lo.count=10,
                                   lo.perc=NULL, 
                                   hi.count=99.9,
                                   hi.perc=NULL,
                                   chunk.size = 1e+06)

print(filtered_myobj)

myobj_normalised <- normalizeCoverage(filtered_myobj)

print(myobj_normalised)

united_df <- unite(myobj_normalised, min.per.group=3L)
print(united_df)

percentage <- percMethylation(united_df, rowids=TRUE)
print(percentage)

methylation_df <- as.data.frame(percentage)
print(methylation_df)
write.csv(methylation_df, "/data/scratch/bt22912/prac/processed_healthyvbenign.csv", row.names = TRUE)

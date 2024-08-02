# install.packages("tm") # requires R 3.3.1 or later
# install.packages("slam")
# install.packages("SnowballC")
# install.packages("stringr")
# install.packages(c("igraph", "igraphdata"))

library(igraph)
library(igraphdata)
library(stringr)
library(slam) # for matrices and arrays
library(tm)
library(SnowballC) # for stemming

## Data Conversion
rm(list = ls())

# creates a file path to the txt folder in CorpusAbstarcts
cname = file.path(".", "CorpusAbstracts", "txt") 
cname

# displays the 15 files within the folder
dir(cname) 
# create a Corpus with the directory's source to be the txt file path
docs = Corpus(DirSource((cname)))
# summary of all documents
summary(docs)

# Print the lengths of each document 
doc_lengths <- sapply(docs, function(doc) {
  length(unlist(strsplit(as.character(doc), "\\W+")))
})
doc_lengths

## Data Transformation

# data cleaning by pattern of repeating words from short to long form
toAI <- content_transformer(function(x, pattern) gsub(pattern, "Artificial Intelligence", x))
docs <- tm_map(docs, toAI, 'AI')

toML <- content_transformer(function(x, pattern) gsub(pattern, "Machine Learning", x))
docs <- tm_map(docs, toAI, 'ML')

toDL <- content_transformer(function(x, pattern) gsub(pattern, "Deep Learning", x))
docs <- tm_map(docs, toAI, 'DL')


# Tokenization
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, content_transformer(tolower))


# filter words - remove stop words and white space
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, stripWhitespace)

# filter words - stemming
docs <- tm_map(docs, stemDocument, language = "english")

# remove the approstrophies
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))

# convert to document term matrix
dtm0 <- DocumentTermMatrix(docs)
# get the top most values and top min values count
freq0 <- colSums(as.matrix(dtm0))
length(freq0)
ord0 = order(freq0)
freq0[head(ord0, 30)]
freq0[tail(ord0, 50)]



# freqeuncy of frequencies
head(table(freq0), 10)
tail(table(freq0), 10)

# Remove hyphens
docs <- tm_map(docs, toSpace, "-")
docs <- tm_map(docs, toSpace, "—")
docs <- tm_map(docs, toSpace, "’")
docs <- tm_map(docs, toSpace, '”')
docs <- tm_map(docs, toSpace, '“')

# out of top 50 highest I remove the words that dont seem to have any importance in classification
docs <- tm_map(docs, toSpace, 'can')
docs <- tm_map(docs, toSpace, 'use')
docs <- tm_map(docs, toSpace, 'also')
docs <- tm_map(docs, toSpace, 'like')
docs <- tm_map(docs, toSpace, 'includ')
docs <- tm_map(docs, toSpace, 'data')
docs <- tm_map(docs, toSpace, 'make')
docs <- tm_map(docs, toSpace, 'one')
docs <- tm_map(docs, toSpace, 'will')
docs <- tm_map(docs, toSpace, 'exampl')
docs <- tm_map(docs, toSpace, 'need')
docs <- tm_map(docs, toSpace, 'new')
docs <- tm_map(docs, toSpace, 'may')
docs <- tm_map(docs, toSpace, "’s")
docs <- tm_map(docs, toSpace, 'mani')
docs <- tm_map(docs, toSpace, 'get')
docs <- tm_map(docs, toSpace, 'generat')
docs <- tm_map(docs, toSpace, 'custom')
docs <- tm_map(docs, toSpace, 'much')

# convert to document term matrix
dtm <- DocumentTermMatrix(docs)
dim(dtm) 
dtms <- removeSparseTerms(dtm, 0.25)
dim(dtms)
as.matrix(dtms)

# from the sparse matrix I remove the words that dont seem to have any importance in classification
docs <- tm_map(docs, toSpace, "amount")
docs <- tm_map(docs, toSpace, "around")
docs <- tm_map(docs, toSpace, "come")
docs <- tm_map(docs, toSpace, "even")
docs <- tm_map(docs, toSpace, "first")
docs <- tm_map(docs, toSpace, "help")
docs <- tm_map(docs, toSpace, "just")
docs <- tm_map(docs, toSpace, "mean")
docs <- tm_map(docs, toSpace, "import")
docs <- tm_map(docs, toSpace, "increas")
docs <- tm_map(docs, toSpace, "right")
docs <- tm_map(docs, toSpace, "still")
docs <- tm_map(docs, toSpace, "time")
docs <- tm_map(docs, toSpace, "well")
docs <- tm_map(docs, toSpace, "take")

# docs <- tm_map(docs, toSpace, 'build')
docs <- tm_map(docs, toSpace, 'certain')
docs <- tm_map(docs, toSpace, 'work')
docs <- tm_map(docs, toSpace, 'year')


# convert to document term matrix
dtm <- DocumentTermMatrix(docs)
dim(dtm) 
dtms <- removeSparseTerms(dtm, 0.3)
dim(dtms)
as.matrix(dtms)

# from the sparse matrix I remove the words that dont seem to have any importance in classification
docs <- tm_map(docs, toSpace, "across")
docs <- tm_map(docs, toSpace, "howev")
docs <- tm_map(docs, toSpace, "live")
docs <- tm_map(docs, toSpace, "look")
docs <- tm_map(docs, toSpace, "often")
docs <- tm_map(docs, toSpace, "provid")
docs <- tm_map(docs, toSpace, "thing")
docs <- tm_map(docs, toSpace, "year")
docs <- tm_map(docs, toSpace, "contribut") 
docs <- tm_map(docs, toSpace, "consid")
docs <- tm_map(docs, toSpace, "found")
docs <- tm_map(docs, toSpace, "less")
docs <- tm_map(docs, toSpace, "major")
docs <- tm_map(docs, toSpace, "differ")
docs <- tm_map(docs, toSpace, "everi")  # everyone, every


# convert to document term matrix
dtm <- DocumentTermMatrix(docs)
dim(dtm) 
dtms <- removeSparseTerms(dtm, 0.35)
dim(dtms)
as.matrix(dtms)

# from the sparse matrix I remove the words that dont seem to have any importance in classification
docs <- tm_map(docs, toSpace, "abl")  # able
docs <- tm_map(docs, toSpace, "accord")  # according
docs <- tm_map(docs, toSpace, "avail")  # available
docs <- tm_map(docs, toSpace, "build") ###########################
docs <- tm_map(docs, toSpace, "continu")  # continue
docs <- tm_map(docs, toSpace, "effect") # KIV
docs <- tm_map(docs, toSpace, "good")
docs <- tm_map(docs, toSpace, "keep")  
docs <- tm_map(docs, toSpace, "larg")  # large
docs <- tm_map(docs, toSpace, "part")  # part of ...
docs <- tm_map(docs, toSpace, "place") ###########################
docs <- tm_map(docs, toSpace, "put")
docs <- tm_map(docs, toSpace, "set")
docs <- tm_map(docs, toSpace, "start")
docs <- tm_map(docs, toSpace, "three")
docs <- tm_map(docs, toSpace, "within")
docs <- tm_map(docs, toSpace, "avoid")
docs <- tm_map(docs, toSpace, "type")
docs <- tm_map(docs, toSpace, "creat")
docs <- tm_map(docs, toSpace, "day")
docs <- tm_map(docs, toSpace, "might")
docs <- tm_map(docs, toSpace, "number")
docs <- tm_map(docs, toSpace, "see")
docs <- tm_map(docs, toSpace, "quick")


# convert to document term matrix
# from the sparse matrix there is very small amount of tokens so none are removed
dtm <- DocumentTermMatrix(docs)
dim(dtm) 
dtms <- removeSparseTerms(dtm, 0.4)
dim(dtms)
as.matrix(dtms)


# convert to document term matrix
dtm <- DocumentTermMatrix(docs)
dim(dtm) 
dtms <- removeSparseTerms(dtm, 0.45)
dim(dtms)
as.matrix(dtms)

# from the sparse matrix I remove the words that dont seem to have any importance in classification
docs <- tm_map(docs, toSpace, "est")
docs <- tm_map(docs, toSpace, "high")
docs <- tm_map(docs, toSpace, "know")
docs <- tm_map(docs, toSpace, "long")
docs <- tm_map(docs, toSpace, "requir")
docs <- tm_map(docs, toSpace, "whether")
docs <- tm_map(docs, toSpace, "sinc")
docs <- tm_map(docs, toSpace, "adapt")
docs <- tm_map(docs, toSpace, "opportun")
docs <- tm_map(docs, toSpace, "addit")  # in addition
docs <- tm_map(docs, toSpace, "better")
docs <- tm_map(docs, toSpace, "now")
docs <- tm_map(docs, toSpace, "big")
docs <- tm_map(docs, toSpace, "though")
docs <- tm_map(docs, toSpace, "anoth")
docs <- tm_map(docs, toSpace, "decis")
docs <- tm_map(docs, toSpace, "follow")
docs <- tm_map(docs, toSpace, "imag")
docs <- tm_map(docs, toSpace, "posit")
docs <- tm_map(docs, toSpace, "possibl")
docs <- tm_map(docs, toSpace, "reli")
docs <- tm_map(docs, toSpace, "sinc")
docs <- tm_map(docs, toSpace, "toward")
docs <- tm_map(docs, toSpace, "among")
docs <- tm_map(docs, toSpace, "sever")
docs <- tm_map(docs, toSpace, "world")
docs <- tm_map(docs, toSpace, "reduc")
docs <- tm_map(docs, toSpace, "light")
docs <- tm_map(docs, toSpace, "key")
docs <- tm_map(docs, toSpace, "becom")
docs <- tm_map(docs, toSpace, "base")
docs <- tm_map(docs, toSpace, "social")
docs <- tm_map(docs, toSpace, "focus")
docs <- tm_map(docs, toSpace, "public")

# convert to final document term matrix
dtm <- DocumentTermMatrix(docs)
dim(dtm) 
dtms <- removeSparseTerms(dtm, 0.45)
dim(dtms)
inspect(dtms)
as.matrix(dtms)

## Text Analysis with Hierarchical Clustering

# Save the dtm matrix to a CSV file
dtms = as.matrix(dtms)
write.csv(dtms, "dtms.csv")

# Compute the cosine distance matrix for the document-term matrix and plot dendogram
distmatrix = proxy::dist(dtms, method = "cosine")
fit = hclust(distmatrix, method = "ward.D")
plot(fit, hang = -1, main = "Abstracts Cosine Distance")

# Create vector of topic labels in same order as corpus
topics = c("adv", "adv", "adv", "ai", "ai", "ai", "clim", "clim", "clim", "food", "food", "food", "sav", "sav", "sav")

# Cut the dendogram to create the required number of clusters (k = 5) and plot
groups = cutree(fit, k = 5)
rect.hclust(fit, k=5, border=2:6)

# Create a table of topic labels vs cluster numbers
clust_table = table(GroupNames = topics, Clusters = groups)
print(clust_table)

# convert table to data frame and prints the accuarcy
TA = as.data.frame.matrix(table(GroupNames = topics, Clusters = groups))
TA

# Calculate the accuracy as the sum of the diagonal elements of the table divided by the total number of documents
accuracy = (TA[1,1] + TA[2,2] + TA[3,3] + TA[4,4] + TA[5,5]) / 15 * 100
print(accuracy)

## Text Analysis with Single Mode Network (Document)

# Base Graph
# Convert the document-term matrix to a matrix
dtmsx = as.matrix(dtms)

# Compute the product of the matrix and its transpose to get the adjacency matrix and remove self-loops
ByAbsMatrix = dtmsx %*% t(dtmsx)
diag(ByAbsMatrix) = 0

# Create an undirected graph from the adjacency matrix with weights
set.seed(32885741)
ByAbs = graph_from_adjacency_matrix(ByAbsMatrix, mode = "undirected", weighted = TRUE)
plot(ByAbs)

# Calculate the degree, closeness and eigenvector of each vertex in dataframe
degree = degree(ByAbs)
closeness = format(closeness(ByAbs), digits = 4)
eig = format(as.table(evcent(ByAbs)$vector), digits = 4)
ksum = as.data.frame(cbind(degree, closeness, eig))
ksum


# Improved Graph
# Create an undirected graph from the adjacency matrix with weights (base graph)
set.seed(32885741)
ByAbs = graph_from_adjacency_matrix(ByAbsMatrix, mode = "undirected", weighted = TRUE)

# Set the size of each vertex  to its eigenvector centrality
V(ByAbs)$size <- eigen_centrality(ByAbs)$vector*20

# Set the width of each edge to its weight
E(ByAbs)$width <- E(ByAbs)$weight*0.01

# Perform community detection using the fast greedy algorithm and plot
cfb = cluster_fast_greedy(ByAbs)
plot(cfb, ByAbs,vertex.label=V(ByAbs)$role,main="Fast Greedy")


## Text Analysis with Single Mode Network (Token)

# Base Graph
# Convert the document-term matrix to a matrix
dtmsa = as.matrix(dtms)

# Compute the product of the matrix and its transpose to get the adjacency matrix and remove self-loops
ByTokenMatrix = t(dtmsa) %*% dtmsa
diag(ByTokenMatrix) = 0

# Create an undirected graph from the adjacency matrix with weights
set.seed(32885741)
ByToken = graph_from_adjacency_matrix(ByTokenMatrix, mode = "undirected", weighted = TRUE)
plot(ByToken)

# Calculate the degree, closeness and eigenvector of each vertex in dataframe
degree = degree(ByToken)
closeness = format(closeness(ByToken), digits = 4)
eig = format(as.table(evcent(ByToken)$vector), digits = 4)
ksum = as.data.frame(cbind(degree, closeness, eig))
ksum

# Improved Graph
set.seed(32885741)
ByToken = graph_from_adjacency_matrix(ByTokenMatrix, mode = "undirected", weighted = TRUE)

# Set the size of each vertex  to its eigenvector centrality
V(ByToken)$size <- eigen_centrality(ByToken)$vector*30

# Set the width of each edge to its weight
E(ByToken)$width <- E(ByToken)$weight*0.02


# plot the graph of vertices and edges updated again
plot(ByToken)

# Perform community detection using the fast greedy algorithm and plot
cfb = cluster_fast_greedy(ByToken)
plot(cfb, ByToken,vertex.label=V(ByToken)$role,main="Fast Greedy")



## Text Analysis with Bipartite Network 
# Convert the document-term matrix to a data frame
dtmsa = as.data.frame(dtms) # clone dtms

# Add row names as a new column named 'ABS' (Abstract)
dtmsa$ABS = rownames(dtmsa) # add row names

# Loop through each row and column of the data frame and create a new row with the weight, abstract, and token information
dtmsb = data.frame()
for (i in 1:nrow(dtmsa)){
  for (j in 1:(ncol(dtmsa)-1)){
    touse = cbind(dtmsa[i,j], dtmsa[i,ncol(dtmsa)],
                  colnames(dtmsa[j]))
    dtmsb = rbind(dtmsb, touse ) } } # close loops

# Rename the columns of dtmsb
colnames(dtmsb) = c("weight", "abs", "token")

# Filter out rows with a weight of 0 and rearrange the columns
dtmsc = dtmsb[dtmsb$weight != 0,] # delete 0 weights
dtmsc = dtmsc[,c(2,3,1)]
dtmsc

# Create a Bipartite plot graph from the data frame with edge weights
set.seed(32885741)
g <- graph.data.frame(dtmsc, directed=FALSE)

# Check the bipartite mapping of the graph and set it as the type of the graph
bipartite.mapping(g)
V(g)$type <- bipartite_mapping(g)$type

# Set vertex color and shape based on type: lightgreen/pink or circle/square
V(g)$color <- ifelse(V(g)$type, "lightgreen", "pink")
V(g)$shape <- ifelse(V(g)$type, "circle", "square")

# Set the edge color to black
E(g)$color <- "black"

# plot the base graph
plot(g)

# Calculate the degree, closeness and eigenvector of each vertex in dataframe
degree = degree(g)
closeness = format(closeness(g), digits = 4)
eig = format(as.table(evcent(g)$vector), digits = 4)
ksum = as.data.frame(cbind(degree, closeness, betweenness, eig))
ksum


# Improved Graph
set.seed(32885741)
g <- graph.data.frame(dtmsc, directed=FALSE)
bipartite.mapping(g)
V(g)$type <- bipartite_mapping(g)$type
V(g)$color <- ifelse(V(g)$type, "lightgreen", "pink")
V(g)$shape <- ifelse(V(g)$type, "circle", "square")
E(g)$color <- "black"

# Set the vertex size to its degree
V(g)$size <- degree(g)

# Set the edge width to its weight
E(g)$width <- E(g)$weight

# Compute the layout for a bipartite graph and swap the columns
LO = layout_as_bipartite(g)
LO = LO[,c(2,1)]

# Scale the bipartite layout by the scaling factor
scaling_factor <- 3
layout_bipartite <- layout_as_bipartite(g)
layout_bipartite_scaled <- layout_bipartite * scaling_factor

# Community detection using the fast greedy algorithm and plot the graph
cfb = cluster_fast_greedy(g)
plot(cfb, g,vertex.label=V(g)$role,main="Fast Greedy")


# Text Analysis Project

## Overview
This project performs text analysis using hierarchical clustering and network analysis to identify similarities and relationships between documents and tokens. The analysis includes dendrograms and network graphs to visualize the results.

## Key Features
- **Hierarchical Clustering**: Uses cosine similarity to measure the angle between document vectors and hierarchical clustering with the `ward.D` method.
- **Document Network Analysis**: Creates a document vs. document matrix and plots an adjacency matrix graph.
- **Token Network Analysis**: Generates a token vs. token matrix and visualizes it with an adjacency matrix graph.
- **Bipartite Network Analysis**: Constructs a bipartite network of documents and tokens and plots the graph with distinct node types.
- **Community Detection**: Implements fast greedy community detection to identify clusters within the networks.
- **Eigenvector Centrality**: Calculates eigenvector scores to determine the importance of nodes in the network.

## Methods

### Hierarchical Clustering
- **Cosine Similarity**: Measures the angle between document vectors.
- **Distance Matrix**: Created using the `proxy` package with cosine method.
- **Clustering**: Hierarchical clustering with the `ward.D` method, visualized with a dendrogram.

### Network Analysis

#### Document Network
- **Matrix Conversion**: DTM converted to document vs. document matrix.
- **Adjacency Matrix Graph**: Plots documents as nodes with edges representing shared terms.

#### Token Network
- **Token Matrix**: Token vs. token matrix created from DTM.
- **Adjacency Matrix Graph**: Plots tokens as nodes with edges representing shared documents.

#### Bipartite Network
- **Data Frame Conversion**: DTM converted to document-token pairs.
- **Graph Plotting**: Documents and tokens plotted as different node types.

## Achievements
- **Accurate Clustering**: Successfully identified distinct clusters for categories like Advertisements, Climate Change, and Food, demonstrating the effectiveness of the cosine similarity and hierarchical clustering approach.
- **Network Visualization**: Created clear and informative network graphs that highlight the relationships and similarities between documents and tokens.
- **Community Detection Insights**: Revealed strong and weak clusters using community detection algorithms, providing deeper insights into the structure of the document and token networks.
- **High Eigenvector Scores**: Identified key documents and tokens with high eigenvector scores, indicating their central role in the network.

## Running the Analysis
1. **Prepare Data**: Place input documents from Corpus Abstract directory in the `data/` directory.
2. **Run Scripts**: Execute R scripts in the `scripts/` directory.
3. **View Results**: Check the `results/` directory for output plots and graphs.


# NSF Grant Data Analysis

This repository contains tools and analyses for processing and understanding NSF (National Science Foundation) grant data from 2019-2025.

## Tools

### 1. [TF-IDF, Semantic Similarity, and Changepoint Analysis](tfidf_semantic_changepoint/README.md)

Analysis package that examines NSF grant abstracts using TF-IDF analysis, semantic similarity metrics, and Bayesian changepoint detection to identify shifts in research language and topics over time.

**Features:**
- TF-IDF analysis of DEI-related keywords
- Semantic similarity using sentence embeddings
- Bayesian changepoint detection to identify structural breaks
- Division-specific and pooled analyses

### 2. [LDA Topic Modeling and Subtopic Clustering](lda_modeling/README.md)

A complete pipeline for analyzing NSF grant abstracts using Latent Dirichlet Allocation (LDA) topic modeling and semantic subtopic clustering to identify research trends across divisions and over time.

**Features:**
- Automatic topic discovery from grant abstracts
- Division-specific topic modeling
- Semantic clustering of related topics
- Yearly trend visualization (2019-2025)

### 3. [NSF Abstract Rewriter](rewrite_nsf_abstract/README.md)

A Gradio web application that rewrites NSF grant abstracts by replacing banned words with neutral, scientific alternatives while preserving all factual content and maintaining the original structure and style.

**Features:**
- Automated word replacement using Google Gemini API
- Preserves scientific accuracy and tone
- Web-based interface

## Repository Structure

```
NSF/
├── tfidf_semantic_changepoint/      # TF-IDF and changepoint analysis
├── lda_modeling/                    # LDA topic modeling pipeline
├── rewrite_nsf_abstract/            # Abstract rewriting web app
├── data/                             # Grant data and outputs
├── models/                           # NLP models and word lists
├── notebooks/                        # Exploratory analysis notebooks
└── screenshots/                      # Visualization examples
```

## Data

Grant data from 2019-2025 is stored in the `data/` directory, organized by year. Processed outputs include:
- Topic assignments and distributions
- Clustered subtopic analyses
- Yearly trend visualizations


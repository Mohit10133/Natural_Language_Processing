# Natural Language Processing Projects

This repository contains a collection of Natural Language Processing (NLP) projects implemented in Python using libraries such as NLTK, spaCy, and Streamlit. Each project focuses on different aspects of NLP techniques and applications.

## Project Structure

```
├── NLP_Preprocessing_techniques/
│   └── NLP_Preprocessing_techniques.py
├── Sparse_vector_embedding/
│   └── Sparse_vector_embedding.py
├── Word_Sense_Disambiguation/
│   └── Word Sense Disambiguation.py
```

## Projects Overview

### 1. NLP Preprocessing Techniques
Located in `NLP_Preprocessing_techniques/NLP_Preprocessing_techniques.py`

An interactive Streamlit application that demonstrates three fundamental NLP tasks:
- **N-gram Language Model**: Trains on the Brown Corpus to calculate word probabilities and sentence perplexity
- **Part-of-Speech (POS) Tagging**: Uses spaCy for high-accuracy POS tagging
- **Named Entity Recognition (NER)**: Identifies named entities like persons, organizations, and locations in text

Features:
- Interactive web interface using Streamlit
- Utilizes both NLTK and spaCy libraries
- Includes bigram probability calculations
- Provides perplexity scores for sentence evaluation
- Displays POS tags and NER results in user-friendly format

### 2. Sparse Vector Embedding
Located in `Sparse_vector_embedding/Sparse_vector_embedding.py`

A Streamlit application focused on text representation and similarity measures:
- **TF-IDF & Euclidean Normalization**: Compute and visualize TF-IDF scores
- **Cosine Similarity**: Find similar words and documents
- **Pointwise Mutual Information (PMI)**: Calculate word association strengths

Features:
- Interactive term frequency and IDF value editing
- Document similarity scoring
- Word vector representations
- Bigram PMI calculations and visualization

### 3. Word Sense Disambiguation
Located in `Word_Sense_Disambiguation/Word Sense Disambiguation.py`

An application that demonstrates various NLP techniques:
- **Positional Indexing**: Create and query positional indexes
- **Word/Document Matrix**: Generate term-document matrices
- **Text Processing**: Implement tokenization, stemming, and lemmatization
- **Edit Distance**: Calculate Levenshtein distance with full trace
- **HMM POS Tagging**: Simple Hidden Markov Model for POS tagging
- **Word Sense Disambiguation**: Implementation of the Lesk algorithm

Features:
- Multiple interactive tabs for different functionalities
- Visualization of word relationships
- Document indexing and search capabilities
- Advanced text processing algorithms

## Requirements

- Python 3.x
- NLTK
- spaCy (with models: en_core_web_lg or en_core_web_sm)
- Streamlit
- Pandas
- NumPy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mohit10133/Natural_Language_Processing.git
```

2. Install required packages:
```bash
pip install nltk spacy streamlit pandas numpy scikit-learn
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_lg
# or for smaller model
python -m spacy download en_core_web_sm
```

4. Download NLTK data:
```python
import nltk
nltk.download(['brown', 'punkt', 'universal_tagset', 'treebank'])
```

## Usage

Each project can be run using Streamlit. Navigate to the project directory and run:

```bash
streamlit run <project_name.py>
```

For example:
```bash
streamlit run NLP_Preprocessing_techniques/NLP_Preprocessing_techniques.py
```

## Features

- Interactive web interfaces for all projects
- Real-time processing and visualization
- Comprehensive NLP pipeline implementations
- Educational examples and demonstrations
- User-friendly input/output interfaces

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

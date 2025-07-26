# Text Classification Project: News Topic Categorizer

## Overview

This project presents a robust pipeline for automatic classification of news articles into topical categories, leveraging advanced NLP techniques and deep learning. The repo demonstrates best practices in data preprocessing, augmentation, model design, and deployment—all designed for scalability and production-readiness.

## Dataset

The dataset comprises a total of 2,225 textual samples collected for the purpose of multi-class news topic classification. These samples are systematically divided into two main subsets: 80% of the data is allocated for training the classification model, enabling it to learn underlying patterns and features across the diverse news topics, while the remaining 20% is reserved for testing, which serves to evaluate the model's generalization performance on unseen data.

The dataset encompasses five distinct classes representing different news categories: business, tech, entertainment, politics, and sport. Each class includes a variety of news articles related to its domain, offering a rich and diverse collection of text samples that aid the model in differentiating between nuanced topic-specific language.


## Methodology

### Data Preprocessing \& Augmentation

- **Data Cleaning:**
    - Removed all full stops and numerals for dataset normalization.
    - Tokenized paragraphs by whitespace.
    - Removed stopwords using NLTK's curated English stopword list for enhanced signal extraction.
- **Balancing:**
    - Employed Easy Data Augmentation (EDA) techniques such as synonym replacement to synthesize samples and address class imbalance efficiently.
- **Label Encoding:**
    - Transformed class labels using Scikit-learn's `LabelEncoder` for model compatibility.


### Vocabulary \& Tokenization

- Identified unique words to construct a dynamic vocabulary.
- Handled OOV (Out Of Vocabulary) and padding:
    - Index 0 assigned for padding.
    - Index 1 reserved for unknown tokens.
- Generated word-to-index mappings for fast lookup and batching.
- Implemented efficient batching with PyTorch's `pad_sequences` and `pack_padded_sequences` for variable-length input handling.


### Model Architecture

- **Embedding Layer:** Dimension size 256, enabling rich word representations.
- **LSTM Layer:** Single-layer, 16 hidden units, excels at capturing contextual nuances.
- **Fully Connected Layer:** Projects to the class scores.
- **Regularization:** L2 (weight decay) set to 1e-4 promotes generalization.
- **Optimizer:** Adam (learning rate 0.001).


### Training Protocol

| Hyperparameter | Value |
| :-- | :-- |
| Batch Size | 32 |
| Epochs | 50 |
| Embedding Size | 256 |
| Hidden Units | 16 |
| LSTM Layers | 1 |
| Weight Decay | 1e-4 |

## Tools \& Frameworks

- **Programming Language:** Python
- **Containerization:** Docker—for seamless portability and reproducible environments.
- **Core Libraries:**
    - PyTorch—for deep learning model implementation.
    - Pandas—for data wrangling and preprocessing.
    - NLTK—for NLP preprocessing.
    - Scikit-learn—for transformations and evaluation.
    - FastAPI—for building scalable RESTful APIs enabling model inference and deployment.


## Standout Features

- **End-to-End Pipeline:** From data ingestion through cleaning, augmentation, modeling, and inference API.
- **Balance \& Fairness:** EDA-powered data balancing minimizes bias.
- **Clean Code:** Modular, well-documented codebase with Dockerized deployment scripts and testing stubs.
- **Ready for Production:** FastAPI API facilitates easy model serving and scaling.


## Quickstart

1. **Clone repository:**

```bash
git clone <repo_url>
```

2. **Build Docker container:**

```bash
docker build -t text-classifier .
docker run -p 8000:8000 text-classifier
```

3. **Predict via API:** Send a POST request to `/predict` with your news text. or go to [localhost](http://localhost:8000/docs)

## Why This Work Stands Out

- Tackles real business challenges in NLP with advanced augmentation and deep learning.
- Codebase designed for maintainability and fast experimentation.
- Easily adaptable for other text classification tasks and datasets.

*For more information, please review the source code and documentation provided in this repository.*


# Sentiment Analysis of Movie Reviews on IMDB Dataset

## Overview

This project uses the IMDB movie review dataset to build a sentiment analysis model. The goal is to classify movie reviews as either positive or negative using word embeddings (Word2Vec and GloVe) and a deep learning model built with TensorFlow/Keras.

### Key Features:
- **Text Preprocessing**: Tokenization and padding of reviews.
- **Word Embeddings**: Uses pre-trained embeddings (Word2Vec and GloVe) for feature representation.
- **Model**: A deep neural network for binary classification (positive/negative sentiment).
- **Evaluation**: Compare the performance of different embeddings (Countries Wiki and GloVe).

## Dataset

The dataset used in this project is the IMDB movie review dataset. You can download it from the following link:

- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

The dataset is split into:
- **Training Set**: Reviews with labeled sentiment (positive/negative).
- **Test Set**: Reviews to evaluate the model performance.

## Requirements

Before running the project, install the required libraries:

```bash
pip install gensim numpy pandas matplotlib scikit-learn tensorflow
```

## Steps

### 1. **Data Preprocessing**

The text data is loaded from the IMDB dataset and preprocessed:
- **Tokenization**: Text is converted to sequences of integers.
- **Padding**: Sequences are padded to ensure uniform input length.

### 2. **Model Training**

Two models are trained using different word embeddings:
- **Countries Wiki Embeddings**: Trained on country-related data from Wikipedia.
- **GloVe Embeddings**: Trained on large corpora like Wikipedia and Gigaword.

Each model uses an embedding layer, followed by a neural network with:
- **GlobalAveragePooling1D**: To capture the semantic meaning of the sequence.
- **Dense Layers**: To predict the sentiment (positive/negative).

### 3. **Model Evaluation**

Both models are evaluated on the validation data, and the validation accuracies are plotted for comparison.

### 4. **Predictions**

You can use the trained model to predict the sentiment of new input text. Here's how the prediction process works:
- Tokenize and pad the input text.
- Use the trained model to predict the sentiment.
- The model will output the sentiment as either **Positive** or **Negative**.

### 5. **Example Prediction**

Example input text for prediction:
```text
"I wish I knew what to make of a movie like this... It seems to be divided into two parts..."
```
The model will classify the sentiment of the review as either **Positive** or **Negative** based on the training data.

### 6. **Model Performance**

- **Countries Wiki Model**: Achieved **78.38%** training accuracy and **73.22%** validation accuracy.
- **GloVe Model**: Achieved **84.78%** training accuracy and **83.18%** validation accuracy.

### 7. **Conclusion**

The model using **GloVe embeddings** performed better due to the larger training corpus used for the embeddings. The model trained on **Countries Wiki embeddings** performed moderately, achieving good accuracy but lower than the GloVe-based model.

## Files

- **train_data**: Preprocessed training data.
- **test_data**: Preprocessed test data.
- **glove.6B.300d.txt**: Pre-trained GloVe embeddings (download from [Stanford GloVe Project](http://nlp.stanford.edu/data/glove.6B.zip)).
- **wiki-countries.w2v**: Pre-trained Countries Wiki embeddings (download from [Countries Word Embedding GitHub](https://github.com/mdeff/countries)).

## References

- [IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- [Stanford GloVe Project](http://nlp.stanford.edu/data/glove.6B.zip)
- [Countries Word Embedding GitHub](https://github.com/mdeff/countries)

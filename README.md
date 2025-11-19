# Sentiment-Analysis-using-Tweets

## Project Overview
This project builds a machine learning pipeline to perform binary classification on tweets. The objective is to identify whether a tweet is reporting a real disaster (Target `1`) or is a non-disaster metaphor/joke (Target `0`). The project compares traditional Recurrent Neural Networks (RNNs) against Convolutional Neural Networks (CNNs) and state-of-the-art Transfer Learning models (USE and BERT).

## Dataset
The dataset is sourced from the Kaggle competition "Natural Language Processing with Disaster Tweets".

* **Training Data:** Contains `text`, `keyword`, `location`, and the `target` label.
* **Class Balance:** The dataset is relatively balanced with **4,342 non-disaster tweets** and **3,271 disaster tweets**.
* **Tweet Length:** The average tweet length in the training set is approximately **15 tokens**.

## Dependencies
To run this notebook, you will need the following libraries:

* `tensorflow`
* `tensorflow_hub` (for Transfer Learning models)
* `tensorflow_text` (for BERT preprocessing)
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`

## Preprocessing
The preprocessing pipeline implemented in the notebook includes:

1.  **Shuffling & Splitting:** Data is shuffled and split into 90% training and 10% validation sets.
2.  **Text Vectorization:**
    * **Vocabulary size:** Capped at 10,000 words.
    * **Sequence length:** Fixed at 15 tokens.
3.  **Embedding:** A standard Keras `Embedding` layer is used for custom models to map integers to dense vectors.

## Models Implemented

### 1. LSTM (Long Short-Term Memory)
A baseline RNN model utilizing an Embedding layer followed by an LSTM layer with 128 units.

### 2. Modified LSTM (with Regularization)
An improvement on the first model, introducing **Dropout (0.2)** and **Early Stopping** to combat overfitting observed in the baseline.

### 3. GRU (Gated Recurrent Unit)
Replaces the LSTM layer with a GRU layer (128 units), offering a simpler architecture with comparable performance.

### 4. CNN (1D Convolutional Neural Network)
Treats text as a sequence signal, utilizing `Conv1D` filters (kernel size 3) and `GlobalAveragePooling1D` to extract features.

### 5. Transfer Learning: Universal Sentence Encoder (USE)
Leverages the pre-trained USE model from TensorFlow Hub (`universal-sentence-encoder/2`) to generate static sentence embeddings, which are fed into a dense classification head.

### 6. Transfer Learning: BERT
Uses the heavy-weight `bert_en_uncased_L-12_H-768_A-12` encoder from TensorFlow Hub.
* **Preprocessing:** Uses the specific BERT preprocessor from TF Hub.
* **Architecture:** `BERT Output` -> `Dense (256, ReLU)` -> `Dropout` -> `Dense (128, ReLU)` -> `Dropout` -> `Output (Sigmoid)`.




## Conclusion
The **Modified BERT** model achieved the highest performance (**81.1% accuracy**), demonstrating that large-scale pre-trained transformers significantly outperform models trained from scratch on this dataset size. The Universal Sentence Encoder also performed strongly with lower computational complexity than BERT.

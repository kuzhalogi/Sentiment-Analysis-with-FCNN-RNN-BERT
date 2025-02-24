# Sentiment Analysis on Emotion-Labeled Tweets: A Comparison of FCNN, RNN (LSTM), and BERT Models

This notebook explores the performance of three deep learning models—**Fully Connected Neural Network (FCNN)**, **Recurrent Neural Network (RNN) with LSTM**, and a **fine-tuned Transformer (BERT)**—on a sentiment analysis task using a dataset of tweet texts labeled with emotions such as joy, sadness, fear, and anger. The dataset is split into training, test, and validation sets, with the validation set used to evaluate the final performance of the models and ensure no overfitting on the test data.


## Key Objectives

1. **Train and Evaluate Models**:
   - Implement and train a **Fully Connected Neural Network (FCNN)** as a baseline model.
   - Build and train a **Recurrent Neural Network (RNN)** with LSTM to capture sequential dependencies in the text.
   - Fine-tune a **Transformer-based model (BERT)** from HuggingFace for state-of-the-art performance.

2. **Compare Model Performance**:
   - Evaluate the models using accuracy and confusion matrices.
   - Analyze the strengths and weaknesses of each approach.

3. **Insights and Learnings**:
   - Compare the results of the three models and explain why certain architectures perform better than others.
   - Discuss the impact of model complexity, training time, and interpretability on sentiment analysis tasks.

## Dataset

- The dataset consists of tweet texts labeled with [emotions](./Emotions_dataset/) (e.g., joy, sadness, fear, anger).
- It is divided into **train**, **test**, and **validation** sets to ensure robust evaluation and prevent overfitting.

## Instructions Followed

1. Train a **Fully Connected Neural Network (FCNN)** for baseline performance.
2. Implement a **Recurrent Neural Network (RNN)** with LSTM to handle sequential data.
3. Fine-tune a **Transformer-based model (BERT)** using HuggingFace for advanced text understanding.
4. Compare the models and analyze their performance to determine the best approach for sentiment analysis.

## Outcome

This project provides a comprehensive comparison of traditional and state-of-the-art models for sentiment analysis, [notebook](./notebooks/NLP_final_project.ipynb) offering insights into their applicability for emotion classification in text data .

---

### Repository Structure
- **`/Emotions_dataset`**: Contains the dataset files (train, test, and validation sets).
- **`/notebooks`**: Jupyter notebooks for data preprocessing, model training, and evaluation.

### Requirements
To run the code in this repository, you'll need the following Python libraries:
- TensorFlow/Keras
- HuggingFace Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib


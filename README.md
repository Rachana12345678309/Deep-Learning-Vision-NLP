# Deep Learning: Computer Vision & Natural Language Processing

This repository features hands-on implementations of advanced neural network architectures using TensorFlow and Keras. The project is divided into two main domains: classifying images with CNNs and analyzing text sentiment with RNNs.

## Part 1: Convolutional Neural Networks (CNN)

CNNs are designed to automatically and adaptively learn spatial hierarchies of features from images. This project implements a CNN to classify the CIFAR-10 dataset (60,000 32x32 color images in 10 classes).

### Key Architectural Layers
- **Convolutional Layer:** Uses filters to create feature maps, detecting patterns like edges, textures, and shapes.
- **Pooling (Max Pooling):** Reduces the spatial dimensions (width and height) of the input volume for the next convolutional layer to reduce computation and prevent overfitting.
- **Flattening:** Converts the 2D feature maps into a 1D vector to be processed by fully connected layers.
- **Dense Layer:** The final classification layers that output the probability for each category.



### Performance Analysis
Unlike a standard Feedforward Network (FNN), the CNN preserves the spatial structure of pixels, leading to significantly higher accuracy in image tasks with fewer parameters.

---

## Part 2: Sentiment Analysis with Recurrent Neural Networks (RNN)

In the second half of this project, I explore NLP by building a model to predict the sentiment (positive/negative) of movie reviews.

### Key Components
- **Word Embeddings:** Converting text into dense vectors of real numbers where semantically similar words are mapped to similar points in the vector space.
- **SimpleRNN:** A type of neural network where the output from the previous step is fed as input to the current step, allowing it to "remember" sequence information.
- **Sigmoid Activation:** Used in the final layer for binary classification (Positive vs. Negative).



## Project Objectives

- Implement **Feature Extraction** through automated convolution filters.
- Understand the **Vanishing Gradient Problem** in simple RNNs and how sequences are processed.
- Use **Data Normalization** and **Tokenization** to prepare raw images and text for deep learning models.

## Tech Stack

- **Language:** Python 3
- **Deep Learning Framework:** TensorFlow / Keras
- **Visualization:** Matplotlib, Seaborn
- **Dataset:** CIFAR-10 (Images) & IMDB/Custom (Text)

## How to Run

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/deep-learning-vision-nlp.git
   cd deep-learning-vision-nlp

2. *Install dependencies:*
   pip install tensorflow numpy matplotlib

3. *Open the notebook:*
   jupyter notebook 35_homework_CNN_basic.ipynb

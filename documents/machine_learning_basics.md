# Machine Learning Basics

Source basis:
Google for Developers - Introduction to Machine Learning / Machine Learning Crash Course
https://developers.google.com/machine-learning/

Machine learning is a way of building computer systems that learn patterns from data.

In traditional programming, a developer writes rules that tell a computer exactly what to do. In machine learning, a model is given examples and learns a relationship from those examples.

## A simple example

Imagine that we want a computer to predict whether an email is spam.

We can collect many example emails and label them as:

- spam
- not spam

A machine learning model studies the examples and looks for useful patterns. After training, the model can receive a new email and predict which class it belongs to.

## Training and prediction

Machine learning usually has two important stages.

### Training

During training, the model learns from data. The model adjusts its internal parameters so its predictions become more accurate.

### Prediction

After training, the model can receive new data that it has not seen before and produce a prediction.

## Features and labels

Features are pieces of information used by a model.

For an apartment-price model, features could include:

- size
- number of rooms
- location
- age of the building

The label is the value the model is trying to predict. In this example, the label could be the apartment price.

## Types of machine learning tasks

Regression predicts a number, such as a price or temperature.

Classification predicts a category, such as spam or not spam.

Machine learning can also be used for recommendation systems, clustering similar items, image recognition, language processing, and many other tasks.

## Why data matters

A model learns from its training data. If the data is poor, incomplete, or not representative of the real world, the model may make poor predictions.

Good machine learning therefore depends not only on the algorithm, but also on the quality of the data and the way the system is evaluated.

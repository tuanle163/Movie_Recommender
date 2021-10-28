# Movie Recommendation Engine 

## 1.Introduction
Movie recommendation engine used the MovieLens 100k Dataset as an input for training and recommendation. The engine was build with neural network to recommend movies that have similar rating or content to a user base on their previous ratings.

__DATASET__

+ 100k Ratings: [Link](https://grouplens.org/datasets/movielens/)

However, the dataset I use for this project can be download directly through __tensorflow-datasets API__, using 
```python
import tensorflow-datasets as tfds

ratings = tfds.load(f'movielens/{dataset}-ratings', split='train')
movies = tfds.load(f'movielens/{dataset}-movies', split='train')
```

The algorithm has 2 parts:

+ __PART 1:__ A trained model that can predict an user ratings and output a list of movies. These movies are those that the model recommend to user.
+ __PART 2:__ An algorithm that search through the recommended movies list above and output top 10 movies.

In PART 1, the model has two stages, Retrival and Ranking. More details of these stages will be in the __Algorithm__ section

## 2.Getting Started

This engine will run on streamlit application. The main.py is the streamlit code. There are two ways you can run the main.py.

__Method 1__: Run Directly in Local

1. Open terminal
2. Go to the main.py directory
3. Run this command in the terminal
```terminal
streamlit run main.py
```

## 3.Algorithms

## 4.Performance Results


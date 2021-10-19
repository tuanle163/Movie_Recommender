import os
import pprint
import tempfile
from typing import Dict, Text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import manifold
from time import process_time
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# size = {'100k': 100_000,
#         '1m': 1_000_000,
#         '20m': 20_000_000,
#         '25m': 25_000_000}

size = [100_000,
        1_000_000,
        20_000_000,
        25_000_000]

def load_dataset(dataset='100k', dataframe_convert=False):
    print(f'Start to load {dataset} dataset')
    ratings = tfds.load(f'movielens/{dataset}-ratings', split='train')
    movies = tfds.load(f'movielens/{dataset}-movies', split='train')
    
    # Select the basic features.
    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"],
    })
    movies = movies.map(lambda x: x["movie_title"])

    if dataframe_convert:
        ratings_df = tfds.as_dataframe(ratings)
        movies_df = tfds.as_dataframe(movies)
        movies_df.rename(columns={' ': 'movie_title'}, inplace=True)
        return ratings, movies, ratings_df, movies_df

    return ratings, movies


def prepare_data(ratings, movies, data_size=size[0]):
    # Randomly shuffle data and split between train and test.
    shuffled = ratings.shuffle(data_size, seed=42, reshuffle_each_iteration=False)

    train_set = shuffled.take(int(0.8*data_size))
    val_set = shuffled.skip(int(0.8*data_size)).take(int(0.2*data_size))
    test_set = shuffled.skip(int(0.6*data_size)).take(int(0.2*data_size))

    movie_titles = movies.batch(int(0.1*data_size))
    user_ids = ratings.batch(1000000).map(lambda x: x["user_id"])

    return user_ids, movie_titles, train_set, val_set, test_set

def init_data_pipeline(train_set, test_set, val_set, data_size=size[0]):
    cached_train = train_set.shuffle(data_size).batch(10000).cache()
    cached_test = test_set.batch(5000).cache()
    cached_val = val_set.batch(5000).cache()

    return cached_train, cached_test, cached_val

def search_algorithm(model, movies_dataset, search_algo='bruteForce'):
    if search_algo=='bruteForce':
        bf_index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        bf_index.index_from_dataset(tf.data.Dataset.zip((movies_dataset.batch(128), movies_dataset.batch(128).map(model.movie_model))))
        return bf_index
        
    elif search_algo=='ScaNN':
        scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
        scann_index.index_from_dataset(tf.data.Dataset.zip((movies_dataset.batch(128), movies_dataset.batch(128).map(model.movie_model))))
        return scann_index

    

# Movie Recommendation Engine 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

__VERSION__: 1.0

__DATE__: 28th of Oct -2021

## 1. Introduction
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

In PART 1, the model has two stages, __Retrival__ and __Ranking__. More details of these stages will be in the __Algorithm__ section

## 2. Getting Started

This engine will run on streamlit application. The main.py is the streamlit code. 

__HOW__: Run Directly in Local

1. Open terminal
2. Go to the main.py directory
3. Run this command in the terminal
```terminal
streamlit run main.py
```

## 3. Algorithms
### __3.1. Overview__

![ver_1.0][logo]

[logo]: /media/movies_recommendation_02.jpeg 'Version 1.0'

This the recommendation model is trained on the 100K MovieLens dataset.

### __3.2. 100K MOVIELENS DATASET INFO__

+ Total User ID: __944__
+ Total Ratings: __100.000__
+ Total Movie Titles: __1664__

Other columns in the dataset.

__User Ratings Table__
| Columns | Data types |
| --- | ----------- |
| bucketized_user_age | float64 |
| movies_genres | object |
| movie_id | object |
| movie_title | object |
| raw_user_age | float64 |
| timestamp | int64 |
| user_gender | bool |
| user_occupation_label | int64 |
| user_occupation_text | object |
| user_rating | float64 |
| user_zip_code | object |

### __3.3. DATA PREPROCESSING__

+ 100000 rating datapoints were splitted into train set (60%), validation set (20%) and test set (20%). 

+ Shuffle -> split -> batch

+ No scaling or standardisation applied

### __3.4. FORWARD FEED__

+ __Inputs:__ User ID and Movies Titles
+ __Labels:__ User Ratings

The model take two inputs (user ID and Movie Titles) 

    --> Go through the embedding layer for both 
    --> Concatenate both embedding layers 
    --> Input into a Fully Connected Layer (FCN) 
    --> Apply Sigmoid activation function at the output 
    --> Scale the output into 0.5 to 5.0 range (This is the rating score range).


### __3.5. WEIGHTS UPDATE__

__a. LOSS FUNCTION__

Two types of loss was calculated to update the weights. 

+ __The Retrieval Loss__

+ __The Ranking Loss__

The Retrieval loss is a loss calculated during the Retrieval stage of the model. The Retrieval stage is the stage when model will choose out the top 100, 50, 10 movies that is suitable to user. Then, these movies will be ranking base on the predicted rating score in the Ranking stage.

+ The Retrieval loss function is __Categorical Cross Entropy__.

+ The Ranking loss funciton is __Mean Absolute Error (MAE)__.

The __Total Loss__ = (The Retrieval Loss * The Retrieval Weights) + (The Ranking Loss * The Ranking Weights)

__b. METRICS__

+ For the Retrieval --> __top_k_accuracy__

+ For the Ranking --> __RMSE__

## 4. Performance Results

__Model hyperparameters__

| __Hyperparameters__ | __Values__ |
| --- | ----------- |
| Embedding dimension (user_id) | 35 |
| Embedding dimension (movie_titles) | 35 |
| Dropout percentage | 50% |
| Learning rate | vary from 0.01 to 0.001 |
| retrieval weight | 1.0 |
| ranking weight | 1.0 |

| __Metrics__ | __Results__ |
| --- | ----------- |
| Retrieval top-100 accuracy | 42.2% |
| Retrieval top-50 accuracy | 27.6% |
| Retrieval top-10 accuracy | 8.2% |
| Ranking RMSE | 0.702 |

__Training History (RMSE and loss)__
![a][training_history]

[training_history]: /media/multitask_model_2021_10_18_training.png

__Top K Accuracy__

![a][training_accuracy]

[training_accuracy]: /media/multitask_model_2021_10_18_training_acc.png
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from utils import *

############################################
###########LOAD UNIQUE VALUES###############
############################################

ratings, movies = load_dataset(dataset='100k')

user_ids, movie_titles, train_set, val_set, test_set = prepare_data(ratings, movies, data_size=size[0])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

class MovielensModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    super().__init__()

    embedding_dimension = 35

    # Embedding layers of User and Movie
    # Compute embeddings for users.
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension, embeddings_regularizer='l2', name='user_embedding')
    ])

    # Compute embeddings for movies.
    self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension, embeddings_regularizer='l2', name='movie_embedding')
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      # tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(256, activation="relu"),
      # tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dropout(0.5),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1, activation='sigmoid')
  ], name='ANN_part')

    # Rating task
    # Loss --> Mean Squared Error (MSE)
    # Metric --> Root Mean Squared Error (RMSE)
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    # Retrieval task
    # Metric --> Factorized Top K
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.movie_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    high = 5.0
    low = 0.5
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features['user_id'])
    # print('done user embedding') # For debug only

    # And pick out the movie features and pass them into the movie model.
    movie_embeddings = self.movie_model(features['movie_title'])
    # print('done movie embedding') # For debug only

    return (
        user_embeddings,
        movie_embeddings,
        # Apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.ratings(tf.concat([user_embeddings, movie_embeddings], axis=1))*(high-low) + low,
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("user_rating")

    user_embeddings, movie_embeddings, rating_predictions = self(features)

    # Compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)
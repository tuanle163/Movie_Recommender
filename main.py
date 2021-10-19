from utils import *
import model_class, os
from model_class import *
import matplotlib.pyplot as plt
import streamlit as st

model_path = './models/multitask_model_2021_10_18_batchnorm_02'
model = model_class.MovielensModel(rating_weight=1.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
model.load_weights(model_path)

cached_train, cached_test, cached_val = init_data_pipeline(train_set, test_set, val_set)
loaded_metrics = model.evaluate(cached_test, return_dict=True)

side_menu = ['For Return User','For New User','About the Recommender']
choice = st.sidebar.selectbox('For Return User', side_menu)

if __name__ == '__main__':

    if choice=='For Return User':
        st.title('Hello! Welcome to Movie Recommender')
        st.write('')
        
        user_id = st.text_input('Input User ID:')
        search_algo = st.multiselect('Please choose the searching algorithm.', ['ScaNN','BruteForce'])

        if user_id=='':
            st.write('Waiting for User ID input!')
        elif user_id != '':
            user_id = str(user_id)
            if search_algo==['ScaNN']:
                index = search_algorithm(model, movies_dataset=movies, search_algo='ScaNN')
                scann_scores, scann_titles = index(tf.constant([user_id]))
                st.write(f'Movie Recommendation for {user_id}: {scann_titles[0, :10]}')

            elif search_algo==['BruteForce']:
                index = search_algorithm(model, movies_dataset=movies, search_algo='bruteForce')
                bf_scores, bf_titles = index(tf.constant([user_id]))
                st.write(f'Movie Recommendation for {user_id}: {bf_titles[0, :10]}')

        

        print('='*100)
        print(f"Retrieval top-100 accuracy: {loaded_metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
        print(f"Retrieval top-50 accuracy: {loaded_metrics['factorized_top_k/top_50_categorical_accuracy']:.3f}.")
        print(f"Retrieval top-10 accuracy: {loaded_metrics['factorized_top_k/top_10_categorical_accuracy']:.3f}.")
        print(f"Ranking RMSE: {loaded_metrics['root_mean_squared_error']:.3f}.")
    





    




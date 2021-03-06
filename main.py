from utils import *
import model_class, os
from model_class import *
import matplotlib.pyplot as plt
import streamlit as st

####################################################
##################### SET UP #######################
####################################################
model_path = './models/multitask_model_2021_10_18_batchnorm_02'
model = model_class.MovielensModel(rating_weight=1.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
model.load_weights(model_path)

##### Save the initial number of User ID
total_user_id = ratings_df['user_id'].unique().shape[0]

##### load the data pipeline
cached_train, cached_test, cached_val = init_data_pipeline(train_set, test_set, val_set)

#### Evaluate the model
loaded_metrics = model.evaluate(cached_test, return_dict=True)

side_menu = ['For Return User','For New User','About the Recommender']
choice = st.sidebar.selectbox('For Return User', side_menu)

def clean_columns(row):
    return str(row).split('\'')[1]

if __name__ == '__main__':

    if choice=='For Return User':
        st.title('Hello! Welcome to Movie Recommender')
        st.write('')
        st.image('media/movies_poster.jpeg')
        
        user_id = st.text_input('Input User ID:')
        search_algo = st.selectbox('Please choose the searching algorithm.', ('ScaNN','BruteForce'))

        if user_id=='':
            st.write('Waiting for User ID input!')
        elif user_id != '':
            user_id = str(user_id)

            st.write(f'You Choose: {search_algo}')
            
            if search_algo=='ScaNN':
                s_index = search_algorithm(model, movies_dataset=movies, search_algo='ScaNN')
                s_scores, s_titles = s_index(tf.constant([user_id]))
                st.write(f'Movie Recommendation for user {user_id}')
                print(s_scores)

                s_movies_list = pd.DataFrame(s_titles.numpy().reshape(10), columns=['Movies List'])
                s_movies_list['Predicted Ratings'] = pd.Series(s_scores.numpy().reshape(10))
                s_movies_list['Movies List'] = s_movies_list['Movies List'].apply(clean_columns)
                
                st.dataframe(s_movies_list)

            if search_algo=='BruteForce':
                bf_index = search_algorithm(model, movies_dataset=movies, search_algo='bruteForce')
                bf_scores, bf_titles = bf_index(tf.constant([user_id]))
                st.write(f'Movie Recommendation for user {user_id}')
                print(bf_scores)

                bf_movies_list = pd.DataFrame(bf_titles.numpy().reshape(10), columns=['Movies List'])
                bf_movies_list['Predicted Ratings'] = pd.Series(bf_scores.numpy().reshape(10))
                bf_movies_list['Movies List'] = bf_movies_list['Movies List'].apply(clean_columns)
                
                st.dataframe(bf_movies_list)
            
    
    elif choice=='For New User':
        st.title('Welcome to Movie Recommenders!!!')
        st.write('')

        st.header('Please Input Your Information')

        col1, col2 = st.columns(2)

        with col1:
            user_name = st.text_input('Please Enter your name:')
            
            if user_name != '': 
                new_user_id = total_user_id + 1
                total_user_id += 1
                st.write(f'The new user have an ID: {new_user_id}')
                st.write(f'new total user id number: {total_user_id}')
        with col2:
            st.write('Please rate the movies!')





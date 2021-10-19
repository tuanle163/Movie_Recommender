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

def extract_movie_title(row):
    return str(row).split('\'')[1]

if __name__ == '__main__':

    if choice=='For Return User':
        st.title('Hello! Welcome to Movie Recommender')
        st.write('')
        
        user_id = st.text_input('Input User ID:')
        search_algo = st.selectbox('Please choose the searching algorithm.', ['ScaNN','BruteForce'])

        if user_id=='':
            st.write('Waiting for User ID input!')
        elif user_id != '':
            user_id = str(user_id)
            if search_algo==['ScaNN']:
                index = search_algorithm(model, movies_dataset=movies, search_algo='ScaNN')
                scores, titles = index(tf.constant([user_id]))
                st.write(f'Movie Recommendation for user {user_id}')

                movies_list = pd.DataFrame(titles.numpy().reshape(10), columns=['Movies List'])
                movies_list = movies_list['Movies List'].apply(extract_movie_title)

                st.dataframe(movies_list)

            elif search_algo==['BruteForce']:
                index = search_algorithm(model, movies_dataset=movies, search_algo='bruteForce')
                scores, titles = index(tf.constant([user_id]))
                st.write(f'Movie Recommendation for user {user_id}')

                movies_list = pd.DataFrame(titles.numpy().reshape(10), columns=['Movies List'])
                movies_list = movies_list['Movies List'].apply(extract_movie_title)
    
    elif choice=='For New User':
        st.title('Welcome to Movie Recommenders!!!')
        st.write('')

        st.header('Please Input Your Information')
        col1, col2 = st.column(2)
        with col1:
            user_name = st.text_input('Please Enter your name:')
            
            if user_name != '': 
                new_user_id = total_user_id + 1
                total_user_id += 1
                st.write(f'The new user have an ID: {new_user_id}')
                st.write(f'new total user id number: {total_user_id}')



    
    





    




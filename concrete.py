from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

def predict_rating(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
model = load_model('catboost.pkl')
#model=pickle.load(open('catboost.pkl','rb'))
st.title('Concrete Comprehensive Strength')
st.write('This is a web app to predict the concrete comprehensive strength based on\
        several features that you can see in the sidebar. Please enter the\
        value of each feature. After that, click on the Predict button at the bottom to\
        see the prediction of the model.')
cement=st.sidebar.text_input(label='Cement')
blast_furnace_slag=st.sidebar.text_input(label='Blast Furnace Slag')
fly_ash=st.sidebar.text_input(label='Fly Ash')
water=st.sidebar.text_input(label='Water')
superplasticizer=st.sidebar.text_input(label='Superplasticizer')
coarse_aggregate=st.sidebar.text_input(label='Coarse Aggregate')
fine_aggregate=st.sidebar.text_input(label='Fine Aggregate')
age=st.sidebar.text_input(label='Age')
features={'cement':cement,'blast_furnace_slag':blast_furnace_slag,'fly_ash':fly_ash,'water':water,'superplasticizer':superplasticizer,'coarse_aggregate':coarse_aggregate,'fine_aggregate ':fine_aggregate,'age':age}
features_df  = pd.DataFrame([features])

st.table(features_df)
if st.button('Predict'):
    
    prediction = predict_rating(model, features_df)
    
    st.write(' Based on feature values, the Concrete Comprehensive Strength is '+ str(int(prediction)))


#pred=model.predict([[float(cement),float(blast_furnace_slag),float(fly_ash),float(superplasticizer),float(coarse_aggregate),float(fine_aggregate),int(age),float(fine_aggregate)]])
#if st.button('Predict'):
    
    
    
    #st.write(' Based on feature values, the car star rating is '+ str(int(pred)))                   

import os
import sys
from io import open

# for data and saves
import pandas as pd
import numpy as np
import dill
from PIL import Image # pillow package
import streamlit as st
import numpy as np
import string
import joblib


# for app
import streamlit as st


# paths
path_to_repo = os.path.dirname(os.getcwd())
print(path_to_repo)
path_to_data = os.path.join(path_to_repo, 'Project', 'Veichle_final_data')


# custom package
sys.path.insert(0, os.path.join(path_to_repo, 'src'))
from emlyon.utils import *






#**********************************************************
#*                      functions                         *
#**********************************************************


def display_score(index):
    # compute model prediction
    pred_score = st.session_state.model.predict([st.session_state.X.values[index]])[0]
    pred_score = int(np.exp(pred_score))
    true_score = int(np.exp(session_state.y[index]))

    # display actual and predicted prices
    col_score, col_score = st.beta_columns(2)
    with col_score:
        centerText('real score', thick = 3)
        centerText(str(true_score) , thick = 4)
    with col_pred:
        centerText('estimated score', thick = 3)
        centerText(str(pred_score) , thick = 4)
    return


def display_score_features(index):
    st.subheader('Score features')
    feat0, val0, feat1, val1 = st.columns([3.5, 1.5, 3.5, 1.5])
    row = st.session_state.X.values[index]
    for i, feature in enumerate(st.session_state.X.columns):
        ind = i % 2
        if ind == 0:
            with feat0:
                st.warning(feature)
            with val0:
                st.info(str(row[i]))
        elif ind == 1:
            with feat1:
                st.warning(feature)
            with val1:
                st.info(str(row[i]))
    return



#**********************************************************
#                     main script                         *
#**********************************************************


#st.sidebar.title('Bulldozer _viewer_')
#st.title('Bulldozer _viewer_')

# session state
if 'model' not in st.session_state:
    # validation set given in notebook
    n_valid = 12000

    # load and preprocess data
    data = pd.read_csv(
        os.path.join(path_to_data, 'Veichle_final_data.csv'), 
        low_memory = False, 
        
    )
    data.ghgScoreavg = np.log(data.ghgScoreavg)
    X, y, nas = proc_df(data, 'ghgScoreavg')
    X, y = X[n_valid:], y[n_valid:]

    path_to_model = os.path.join(path_to_repo, 'Project', 'RF_classifier.pk')
    with open(path_to_model, 'rb') as file:
        model = dill.load(file)
        
 



def main():
    st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Greenhouse Gas Rating Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: Black;'>Insert your vheicles features and we will do  the rest.</h3>", unsafe_allow_html=True)
    st.sidebar.header("What is this Project about?")
    st.sidebar.text("It a Web app that would help the user in determining the probabile GHG SCORE based on the feature of the vehicles.")
    st.sidebar.header("What tools where used to make this?")
    st.sidebar.text("The Model was made using a dataset from www.fueleconomy.gov. We made use a random forest classifier.")
    
    

cylinders = st.slider("Cylinder",2,16) 
year = st.slider("year",2013,2022) 
startstop = st.selectbox("Does your car has start stop?",["Yes","No"])
drive = st.selectbox("Tell us the axel type of your car?",["Front-Wheel Drive","All-Wheel Drive","Rear-Wheel Drive","4-Wheel Drive","Part-time 4-Wheel Drive"])
make = st.selectbox("Tell us the manufacturer of your car?",["BMW ","Ford","Chevrolet","Mercedes-Benz","Porsche","Toyota ","GMC","Audi","Nissan","Hyundai","Kia", "Honda", "Jeep","MINI", "Lexus","Volkswagen", "Cadillac ", "Dodge", "Jaguar","Mazda","Infiniti","Subaru", "Volvo ","Lincoln", "Land Rover","Mitsubishi", "Buick", "Acura", "Ram", "Tesla", "Ferrari", "Chrysler ","Maserati", "Genesis", "Bentley  ", "Rolls-Royce ", "Fiat", "Lamborghini", "Aston Martin", "Alfa Romeo", "Scion", "McLaren Automotive ", "Roush Performance", "smart","Lotus", "Suzuki ", "Bugatti", "Lucid", "Polestar", "BYD", "Karma", "Pagani" ])

inputs =[[cylinders,year,startstop,drive,make]]



if st.button('Predict'): #making and printing our prediction
    result = model.predict(inputs)
    updated_res = result.flatten().astype(float)
    st.success('The Probability of getting admission is {}'.format(updated_res))
    
if __name__ =='__main__':
  main() #calling the main method
  



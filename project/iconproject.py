#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:19:27 2022

@author: davide
"""

import streamlit as st 
import numpy as np 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu
from PIL import Image


prostatecancer_model = pickle.load(open('/Users/elettidavide/classificatori/cancerprostate_model.sav','rb'))

parkinsons_model = pickle.load(open('/Users/elettidavide/classificatori/parkinsons_model.sav', 'rb'))

lungcancer_model = pickle.load(open('/Users/elettidavide/classificatori/lungcancer_model.sav', 'rb'))

heartdisease_model = pickle.load(open('/Users/elettidavide/classificatori/heartdisease_model.sav', 'rb'))



st.set_page_config(page_title="HS", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

    
selected = option_menu(
        
        menu_title = 'Health system',
        options = ['---------All dataset used',
                   '---------Parkinsons prediction',
                   '---------Lung Cancer prediction',
                   '---------Heart Disease prediction',
                   '---------Prostate Cancer prediction',
                   '---------Breast Cancer classifier'],
        
        icons=['archive','person','binoculars','heart','filter','gender-female'],
                    
        menu_icon = "cast",
        
        default_index=0,
                          
        orientation="horizontal"
    
    )


if (selected == '---------All dataset used'):
    # Model building
    def build_model(df):
        X = df.iloc[:,:-1] # Using all column except for the last column as X
        Y = df.iloc[:,-1] # Selecting the last column as Y


    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('Upload Dataset'):
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    # Displays the dataset
    st.title("DATASET")
    st.subheader("View of the loaded dataset ⬇️ ")
    if  uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        build_model(df)






if (selected == '---------Breast Cancer classifier'):
    
    st.title("Breast Cancer Classifier")
    
    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
    ) 
    
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    image = Image.open('/Users/elettidavide/classificatori/breastcancer.png')
    st.image(image)
            
    st.write('Shape of dataset:', X.shape)
    st.write('Number of classes:', len(np.unique(y)))

    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
        elif clf_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['n_estimators'] = n_estimators
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        else:
            clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], random_state=1234)
        return clf

    clf = get_classifier(classifier_name, params)
    #### CLASSIFICATION ####

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Accuracy =', acc)
    st.write(f'Classifier = {classifier_name}')
    

    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

    plt.xlabel('xlabel')
    plt.ylabel('ylabel')
    plt.colorbar()

    #plt.show()
    st.pyplot(fig)







# Parkinson's Prediction Page
if (selected == "---------Prostate Cancer prediction"):
    
    # page title
    st.title("Prostate Cancer Prediction")
    st.subheader("Prediction of Prostate Cancer (classifier - SVM)") 
    st.write("Shape of dataset : (100, 9)")
    st.write("1(Benign) : 35")
    st.write("0(Malignant) : 65")
    st.write("Accuracy score of the training data : 0.83")
    st.write("Accuracy score of the test data : 0.85")
        
    
    image = Image.open('/Users/elettidavide/classificatori/prostatecancer.png')
    st.image(image)

    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius = st.text_input('Radius')
        
    with col2:
        texture = st.text_input('Texture')
    
    with col3:
        perimeter = st.text_input('Perimeter')
    
    with col1:
        area = st.text_input('Area')
    
    with col2:
        smoothness = st.text_input('Smoothness')
    
    with col3:
        compactness = st.text_input('Compactness')
    
    with col1:
        symmetry = st.text_input('Symmetry')
    
    with col2:
        fractal_dimension = st.text_input('Fractal dimension')
    
    # code for Prediction
    cancer_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Prostate cancer prediction Result'):
        cancer_prediction = prostatecancer_model.predict([[radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension]])
        
        if (cancer_prediction[0] == 1):
          cancer_diagnosis = 'The tumor is Benign'
        else:
          cancer_diagnosis = 'The tumor is Malignant'
        
    st.success(cancer_diagnosis)
        
    
    
    
    
    
# Parkinson's Prediction Page
if (selected == "---------Parkinsons prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction")
    st.subheader("Prediction of Parkinson's Disease (classifier - SVM)") 
    st.write("Shape of dataset : (195, 24)")
    st.write("1(Parkinson's Positive) : 147")
    st.write("0(Healthy) : 48")
    st.write("Accuracy score of the training data : 0.87")
    st.write("Accuracy score of the test data : 0.87")
    
    image = Image.open('/Users/elettidavide/classificatori/parkinson.png')
    st.image(image)
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo (Hz)')
        
    with col2: 
        fhi = st.text_input('MDVP:Fhi (Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo (Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter (%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter (Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer (dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)







# Lung cancer Prediction Page
if (selected == '---------Lung Cancer prediction'):
    
    # page title
    st.title('Lung Cancer Prediction')
    st.subheader("Prediction of Lung Cancer (classifier - SVM)")
    st.write("Shape of dataset : (309, 16)")
    st.write("1(lung cancer Positive) : 270")
    st.write("0(lung cancer Negative) : 39")
    st.write("Accuracy score of the training data : 0.94")
    st.write("Accuracy score of the test data : 0.90")

    image = Image.open('/Users/elettidavide/classificatori/lungcancer.png')
    st.image(image)
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        AGE = st.text_input('AGE')
        
    with col2:
        SMOKING = st.text_input('SMOKING (2=YES - 1=NO)')
    
    with col3:
        YELLOW_FINGERS = st.text_input('YELLOW FINGERS (2=YES - 1=NO)')
    
    with col1:
        ANXIETY = st.text_input('ANXIETY (2=YES - 1=NO)')
    
    with col2:
        PEER_PRESSURE = st.text_input('PEER PRESSURE (2=YES - 1=NO)')
    
    with col3:
        CHRONIC_DISEASE = st.text_input('CHRONIC DESEASE (2=YES - 1=NO)')
    
    with col1:
        FATIGUE = st.text_input('FATIGUE (2=YES - 1=NO)')
    
    with col2:
        ALLERGY = st.text_input('ALLERGY (2=YES - 1=NO)')

    with col3:
        WHEEZING = st.text_input('WHEEZING (2=YES - 1=NO)')
    
    with col1:
        ALCOHOL_CONSUMING = st.text_input('ALCOHOL COSUMING (2=YES - 1=NO)')   
     
    with col2:
        COUGHING = st.text_input('COUGHING (2=YES - 1=NO)')

    with col3:
        SHORTNESS_OF_BREATH = st.text_input('SHORTNESS OF BREATH (2=YES - 1=NO)')
    
    with col1:
        SWALLOWING_DIFFICULTY = st.text_input('SWALLOWING_DIFFICULTY (2=YES - 1=NO)')   
     
    with col2:
        CHEST_PAIN = st.text_input('CHEST_PAIN (2=YES - 1=NO)')

    # code for Prediction
    lungcancer_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Lung Cancer Test Result'):
        lungcancer_prediction = lungcancer_model.predict([[AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE ,ALLERGY ,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]])
        
        if (lungcancer_prediction[0] == 1):
          lungcancer_diagnosis = 'The Person has lung cancer'
        else:
          lungcancer_diagnosis = 'The Person does not have lung cancer'
        
    st.success(lungcancer_diagnosis)
    
    
    
    
    
    
# Heart Disease Prediction Page
if (selected == '---------Heart Disease prediction'):
    
    # page title
    st.title('Heart Disease Prediction')
    st.subheader("Prediction of Heart Disease (classifier - SVM)")
    st.write("Shape of dataset : (303, 14)")
    st.write("1(Defective Heart) : 165")
    st.write("0(Healthy Heart) : 138")
    st.write("Accuracy score of the training data :0.82")
    st.write("Accuracy score of the test data : 0.85")


    image = Image.open('/Users/elettidavide/classificatori/heartdisease.png')
    st.image(image)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('0 = normal - 1 = fixed defect - 2 = reversable defect')
        
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heartdisease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
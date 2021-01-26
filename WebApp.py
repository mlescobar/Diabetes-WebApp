# This program detects if someone has diabetes

# Import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Create a tittle and a subtitle
st.write(""" 
# Diabetes Detection
Detect if someone has diabetes.
""")

# Open and display an image
image = Image.open('diabetes-icon-4.jpg')
st.image(image, caption='Diabetes')

# Get the data
df = pd.read_csv('diabetes.csv')
# Set a subheader
st.subheader('Data Information: ')
# Show the data as a table
st.dataframe(df)
# SHow stadistics on the data
st.write(df.describe())
# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into independent x and y variables
X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 17, 3)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI= st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    # Store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }

    # Transform the data into a df
    features = pd.DataFrame(user_data, index = [0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and siplay the users input
st.subheader('User Input:')
st.write(user_input)

# Create the train model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,y_train)

# SHow the models metric
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification:    ')
st.write(prediction)
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_note_authentication(df):
    print(df)
    prediction=classifier.predict(df)
    print(prediction)
    return prediction



def main():

    html_temp ="""
    <div style="background-black:Blue;padding:10px">
    <h2 style="color:black;text-align:center;">CUSTOMER CHURN PREDICTION SYSTEM</h2>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    #st.image("Automation-in-telecom-industry.jpg")

    cat_vars = {
    'MultipleLines': ['No phone service', 'Yes', 'No'],
    'InternetService': ['Fiber optic', 'DSL', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    }

   

    # create an empty dataframe with the correct columns
    input_data = pd.DataFrame(columns=[col + '_' + val for col in cat_vars.keys() for val in cat_vars[col]] + ['PaperlessBilling', 'Partner', 'Dependents'])

    # fill in the dataframe with the user inputs
    for key in cat_vars.keys():
      input_data[[key + '_' + val for val in cat_vars[key]]] = np.zeros((1, len(cat_vars[key])))
      input_data[key + '_' + st.selectbox(key, cat_vars[key])] = 1
    #input_data['tenure'] = st.number_input('Tenure (Int number (1 -72))', key='tenure_input', min_value=1, max_value=72, value=1, step=1)
    #input_data['MonthlyCharges'] = st.number_input('MonthlyCharges', key='monthly_charges_input', value=0)
    input_data['PaperlessBilling'] = st.selectbox('PaperlessBilling', key='paperless_billing_input', options=[0, 1])
    input_data['Partner'] = st.selectbox('Partner', key='partner_input', options=[0, 1])
    input_data['Dependents'] = st.selectbox('Dependents', key='dependents_input', options=[0, 1])


    # use the OneHotEncoder to transform the input data
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder()
    input_data_encoded = encoder.fit_transform(input_data).toarray()
    print("input_data----------->",input_data.columns)
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(input_data)
        st.success('The output is {}'.format(result))
        if result == 0:
            st.write("Customer Will Not Churn")
        else:
            st.write("Customer Will Churn")
        
    if st.button("About"):
        st.text("Shreyas, DataScientist Trainee - TuringMinds.ai")

    

if __name__=='__main__':
    main()
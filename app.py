import streamlit as st
import pandas as pd
import pickle

# Load the trained model and necessary data
model = pickle.load(open("C:\\Users\\shekh\\Downloads\\Customer churn analysis Ml Project\\best_model_Gradient_Boosting.pkl", "rb"))
df_1 = pd.read_csv("C://Users//shekh//Downloads//Customer churn analysis Ml Project//first_telc.csv")

# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    # Collect user input
    inputQuery1 = st.text_input("SeniorCitizen")
    inputQuery2 = st.text_input("MonthlyCharges")
    inputQuery3 = st.text_input("TotalCharges")
    inputQuery4 = st.selectbox("Gender", ["Male", "Female"])
    inputQuery5 = st.selectbox("Partner", ["Yes", "No"])
    inputQuery6 = st.selectbox("Dependents", ["Yes", "No"])
    inputQuery7 = st.selectbox("PhoneService", ["Yes", "No"])
    inputQuery8 = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
    inputQuery9 = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    inputQuery10 = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    inputQuery11 = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    inputQuery12 = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
    inputQuery13 = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    inputQuery14 = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
    inputQuery15 = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
    inputQuery16 = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    inputQuery17 = st.selectbox("PaperlessBilling", ["Yes", "No"])
    inputQuery18 = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    inputQuery19 = st.text_input("Tenure")

    # Button to trigger prediction
    if st.button("Predict"):
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
                 inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
                 inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

        new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                             'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                             'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                             'PaymentMethod', 'tenure'])

        # Preprocessing steps
        df_2 = pd.concat([df_1, new_df], ignore_index=True)
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        df_2.drop(columns=['tenure'], axis=1, inplace=True)

        new_df__dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                               'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                               'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

        single = model.predict(new_df__dummies.tail(1))
        probability = model.predict_proba(new_df__dummies.tail(1))[:, 1]

        if single == 1:
            result = "This customer is likely to be churned"
        else:
            result = "This customer is likely to continue"

        st.write(result)
        st.write("Confidence:", probability * 100)

if __name__ == "__main__":
    main()

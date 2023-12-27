import streamlit as st
import pickle
import numpy as np
import pandas as pd 

from sklearn.preprocessing import LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(LabelEncoder().fit_transform)

# Load the trained model & preprocessing Steps
 
loaded_le_model=pickle.load(open('LE_MODEL.pkl', 'rb'))

loaded_pca_fit=pickle.load(open('PCA_MODEL.pkl', 'rb'))

loaded_rf_model=pickle.load(open('RF_MODEL.PKL', 'rb'))

# Define the mushroom classes
classes = ["edible", "poisonous"]

# Define the Streamlit app
def main():
    st.title("Mushroom Classification")
    st.write("This app classifies mushrooms as edible or poisonous.")

    # Gather user input
    gc = st.selectbox("select gill color", ["black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"])
    spc = st.selectbox("spore print color", ["black", "brown", "buff", "chocolate", "green", "orange", "purple", "white", "yellow"])
    p = st.selectbox("population", ["abundant", "clustered", "numerous", "scattered", "several", "solitary"])
    gs = st.selectbox("gill size", ["broad", "narrow"])
    o = st.selectbox("odor", ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])
    b = st.selectbox("bruises", ["bruises", "no"])
    ss = st.selectbox("stalk shape", ["enlarging", "tapering"])
    scar = st.selectbox("stalk color above ring", ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"])
    sr = st.selectbox("stalk root", ["bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"])

    # Add more features here...

    # Prepare the input features
    features = np.array([gc, spc, p, gs, o, b, ss, scar, sr])
    df = pd.DataFrame([features])

    # Make predictions
    prediction = loaded_rf_model.predict(loaded_pca_fit.transform(loaded_le_model.transform(df.head(1))))

    # Display the prediction
    st.write("Prediction:", classes[prediction[0]])

if __name__ == "__main__":
    main()

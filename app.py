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

features=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
classes = {'e':"edible",'p' :"poisonous"}

# Define the Streamlit app
def main():
    st.title("Mushroom Classification")
    st.write("This app classifies mushrooms as edible or poisonous.")

    # # Gather user input
    # gc = st.selectbox("select gill color", ["black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"])
    # spc = st.selectbox("spore print color", ["black", "brown", "buff", "chocolate", "green", "orange", "purple", "white", "yellow"])
    # p = st.selectbox("population", ["abundant", "clustered", "numerous", "scattered", "several", "solitary"])
    # gs = st.selectbox("gill size", ["broad", "narrow"])
    # o = st.selectbox("odor", ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])
    # b = st.selectbox("bruises", ["bruises", "no"])
    # ss = st.selectbox("stalk shape", ["enlarging", "tapering"])
    # scar = st.selectbox("stalk color above ring", ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"])
    # sr = st.selectbox("stalk root", ["bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"])
    # sr = st.selectbox("stalk root", ["bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"])
    # ye apne Black Brown ye kaha se dala? DAta mein tou E d h
    # yes sir data read kia tha or waha sa mja pata chala tha ya color ha
    # Black to Letter Mapping kaha se li? ya nae ha?
    
    labels = {
        'cap-shape': [['x', 'b', 's', 'f', 'k', 'c'], ['Convex', 'Bell', 'Sunken', 'Flat', 'Knobbed', 'Conical']],
        'cap-surface': [['s', 'y', 'f', 'g'], ['Smooth', 'Scaly', 'Fibrous', 'Grooves']],
        'cap-color': [['n', 'y', 'w', 'g', 'e', 'p', 'b', 'u', 'c', 'r'], ['Brown', 'Yellow', 'White', 'Gray', 'Red', 'Pink', 'Buff', 'Purple', 'Cinnamon', 'Green']],
        'bruises': [['t', 'f'], ['True', 'False']],
        'odor': [['p', 'a', 'l', 'n', 'f', 'c', 'y', 's', 'm'], ['Pungent', 'Almond', 'Anise', 'None', 'Foul', 'Creosote', 'Fishy', 'Spicy', 'Musty']],
        'gill-attachment': [['f', 'a'], ['Free', 'Attached']],
        'gill-spacing': [['c', 'w'], ['Close', 'Crowded']],
        'gill-size': [['n', 'b'], ['Narrow', 'Broad']],
        'gill-color': [['k', 'n', 'g', 'p', 'w', 'h', 'u', 'e', 'b', 'r', 'y', 'o'], ['Black', 'Brown', 'Gray', 'Pink', 'White', 'Chocolate', 'Purple', 'Buff', 'Green', 'Red', 'Yellow', 'Orange']],
        'stalk-shape': [['e', 't'], ['Enlarging', 'Tapering']],
        'stalk-root': [['e', 'c', 'b', 'r', '?'], ['Equal', 'Club', 'Bulbous', 'Rooted', 'Missing']],
        'stalk-surface-above-ring': [['s', 'f', 'k', 'y'], ['Smooth', 'Fibrous', 'Silky', 'Scaly']],
        'stalk-surface-below-ring': [['s', 'f', 'y', 'k'], ['Smooth', 'Fibrous', 'Scaly', 'Silky']],
        'stalk-color-above-ring': [['w', 'g', 'p', 'n', 'b', 'e', 'o', 'c', 'y'], ['White', 'Gray', 'Pink', 'Brown', 'Buff', 'Cinnamon', 'Orange', 'Chocolate', 'Yellow']],
        'stalk-color-below-ring': [['w', 'p', 'g', 'b', 'n', 'e', 'y', 'o', 'c'], ['White', 'Pink', 'Gray', 'Buff', 'Brown', 'Cinnamon', 'Yellow', 'Orange', 'Chocolate']],
        'veil-type': [['p'], ['Partial']],
        'veil-color': [['w', 'n', 'o', 'y'], ['White', 'Brown', 'Orange', 'Yellow']],
        'ring-number': [['o', 't', 'n'], ['One', 'Two', 'None']],
        'ring-type': [['p', 'e', 'l', 'f', 'n'], ['Pendant', 'Evanescent', 'Large', 'Flaring', 'None']],
        'spore-print-color': [['k', 'n', 'u', 'h', 'w', 'r', 'o', 'y', 'b'], ['Black', 'Brown', 'Buff', 'Chocolate', 'White', 'Green', 'Orange', 'Yellow', 'Purple']],
        'population': [['s', 'n', 'a', 'v', 'y', 'c'], ['Several', 'Solitary', 'Abundant', 'Numerous', 'Clustered']],
        'habitat': [['u', 'g', 'm', 'd', 'p', 'w', 'l'], ['Urban', 'Grasses', 'Meadows', 'Woods', 'Paths', 'Waste', 'Leaves']]
    }

    def zipper(key,value):
        print(labels.get(key))
        return dict(zip(labels.get(key)[0],labels.get(key)[1])).get(value)

    selection_list=[]
    # Generate Streamlit dropdowns and output as valid Python code
    for key, values in labels.items():
        selection_list.append(st.selectbox(key=key,label=f'Select {key}',options=values,format_func=lambda x: zipper(key,x)))

    # Add more features here...

    # Prepare the input features
    feature_val = [selection_list]
    df = pd.DataFrame(feature_val,columns=features)

    st.write(df)

    if st.button('Predict'):   
        # Make predictions
        prediction = loaded_rf_model.predict(loaded_pca_fit.transform(loaded_le_model.transform(df.head(1))))
    
        # Display the prediction
        st.write("Prediction:", classes.get(prediction[0]))

if __name__ == "__main__":
    main()

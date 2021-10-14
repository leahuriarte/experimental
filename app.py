"remember pipreqs for requirements.txt"


from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

import streamlit as st
import streamlit_analytics




class Predict:
    def __init__(self, filename):

        st.title('What Trash Do You Have? (Recycle?)')

        self.learn_inference = load_learner(Path()/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()

        with streamlit_analytics.track():
            st.text_input(" ",value="Say Why You Think Leah Is Great In This Text Box (Or Else) (Press Enter to Submit)")

        
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Submit Your Photo Below",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Your Image')

    def get_prediction(self):

        if st.button('Classify It!'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            pred = pred.capitalize()
            st.write(f'My Prediction: {pred}')
            percentage = probs[pred_idx]*100
            st.write(f"Probability that I'm Correct: {percentage:.02f}%")
            if pred != "Trash":
                st.write('You can recycle this! Yay! Wow! So Cool!')
            else:
                st.write('It is just regular trash. No recycle, probably :(')
        #else: 
            #st.write(f'Click the button to classify') 

if __name__=='__main__':

    file_name='recycling.pkl'

    predictor = Predict(file_name)
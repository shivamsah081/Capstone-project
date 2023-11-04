import streamlit as st
from io import StringIO
import streamlit_option_menu
import time
import os
import glob
import numpy as np
from gtts import gTTS
from googletrans import Translator
from lstm_model import Data,model
import torch
import pickle

st.header(

    "Indian Sign Language Interpretation Integrating Pose Detection and Speech Synthesis"
)

page_bg_img = '''
<style>
.stApp {
background-image: url("https://wallpaperset.com/w/full/3/d/6/38621.jpg");
bbackground-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#model = LSTM()
#model.load('D:/Praxis/capstone/hand_sign_model.pth')
target = pickle.load(open('target_new.pkl', 'rb'))

caption1 = ''

uploaded_file = st.file_uploader('Choose a file')

if uploaded_file is not None:
    video_path = "D:/Praxis/capstone/" + uploaded_file.name
    video_file = open(video_path, "rb").read()
    #print(video_file)

    st.video(video_file, format='video/mp4')
    data,frames = Data.video_data(video_path)    
    X = Data.prepare_data(data)
    #print(len(X))


    if st.button("predict"):
        #caption = uploaded_file.name.replace('.mp4', '')
        out_index,(_,_) = model(torch.tensor(X).float().to('cpu'))
        out_index = out_index[30].argmax()
        #global prediction
        prediction = target[out_index]
        st.subheader(prediction)
        caption1 = prediction


# with st.sidebar:
#     selected = streamlit_option_menu.option_menu(
#         menu_title = None,
#         options = ['Home', 'Contact', 'Logout'],
#         icons = ['house-door-fill', 'person-lines-fill', 'box-arrow-left']
#     )

# try:
#     os.mkdir("temp")
# except:
#     pass
#st.title("Text to speech")

#text = st.text_input("Enter text")
# in_lang = st.selectbox(
#     "Select your input language",
#     ("English", "Hindi", "Bengali", "korean", "Chinese", "Japanese"),
# )
# if in_lang == "English":
#     input_language = "en"
# elif in_lang == "Hindi":
#     input_language = "hi"
# elif in_lang == "Bengali":
#     input_language = "bn"
# elif in_lang == "korean":
#     input_language = "ko"
# elif in_lang == "Chinese":
#     input_language = "zh-cn"
## elif in_lang == "Japanese":
##     input_language = "ja"
#
#out_lang = st.selectbox(
#    "Select your output language",
#    ("English", "Hindi", "Bengali", "korean", "Chinese", "Japanese"),
#)
#if out_lang == "English":
#    output_language = "en"
#elif out_lang == "Hindi":
#    output_language = "hi"
#elif out_lang == "Bengali":
#    output_language = "bn"
#elif out_lang == "korean":
#    output_language = "ko"
#elif out_lang == "Chinese":
#    output_language = "zh-cn"
#elif out_lang == "Japanese":
#    output_language = "ja"
#
#english_accent = st.selectbox(
#    "Select your english accent",
#    (
#        "Default",
#        "India",
#        "United Kingdom",
#        "United States",
#        "Canada",
#        "Australia",
#        "Ireland",
#        "South Africa",
#    ),
#)
#
#if english_accent == "Default":
#    tld = "com"
#elif english_accent == "India":
#    tld = "co.in"
#
#elif english_accent == "United Kingdom":
#    tld = "co.uk"
#elif english_accent == "United States":
#    tld = "com"
#elif english_accent == "Canada":
#    tld = "ca"
#elif english_accent == "Australia":
#    tld = "com.au"
#elif english_accent == "Ireland":
#    tld = "ie"
#elif english_accent == "South Africa":
#    tld = "co.za"

translator = Translator()

def text_to_speech(text):
    #translation = translator.translate(text, src=input_language, dest=output_language)
    print(type(text))
    translation = translator.translate(text)
    trans_text = translation.text
    print("Word saved in trans_text:", trans_text)
    tts = gTTS(trans_text, slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name, trans_text


#display_output_text = st.checkbox("Display output text")

if st.button("play"):
    #result, output_text = text_to_speech(output_language, uploaded_file.name.replace('.mp4', ''), tld)
    print("What is inside caption1, let's see",prediction)
    result, output_text = text_to_speech(prediction)
    audio_file = open(f"temp/{result}.mp3", "rb")
    audio_bytes = audio_file.read()
    st.markdown(f"## Your audio:")
    st.audio(audio_bytes, format="audio/mp3")

    # if display_output_text:
    #     st.markdown(f"## Output text:")
    #     st.write(f" {output_text}")
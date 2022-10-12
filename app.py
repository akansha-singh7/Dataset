import streamlit as st
#from streamlit import scriptrunner
import tensorflow as tf
import os
import cv2
from PIL import Image, ImageOps
import numpy as np

#Load Model
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/my_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

#Title
st.write("""
         # Hidden Service Analysis
         """
         )

#Upload Image
file = st.file_uploader("Upload the image file here", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)

#Prediction of Image Funtion
def import_and_predict(image_data, model):

        #Size of Image
        size = (200,200)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        #Image to Array
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]

        #Prediction
        prediction = model.predict(img_reshape)
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    #Open Image
    image = Image.open(file)
    st.image(image, use_column_width=True)

    #Call the function to make prediction
    prediction = import_and_predict(image, model)
    score = tf.nn.softmax(prediction[0])

    #Prediction of Class
    if np.argmax(score) == 0:
        st.write("It is a Card!")
    elif np.argmax(score) == 1:
        st.write("It is a Device!")
    elif np.argmax(score) == 2:
        st.write("It is a Hacker!")
    else:
        st.write("It is a Money!")

import codecs
from bs4 import BeautifulSoup
import string
from nltk import tokenize
from operator import itemgetter
import math
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
import re

#Upload HTML file
doc_path = st.file_uploader("Upload the text file here", type=["html"])

if doc_path is None:
    st.text("Please upload the text file")
else:
#Get doc name
  doc = doc_path.name

#Copy file content
  with open(os.path.join("/content/", doc),"wb") as f:
                f.write((doc_path).getbuffer())

#Read the file
  file=open(doc,"r", 1)
  document= BeautifulSoup(file.read()).get_text()

#Remove punctuation
  remove = string.punctuation
  remove = remove.replace(".", "")
  pattern = r"[{}]".format(re.escape(remove))
  document = re.sub(pattern, "", document)

#Lower class
  filtered = document.translate(str.maketrans('', '', string.punctuation))
  filtered = document.lower()

#Get total word length
  total_words = filtered.split()
  total_word_length = len(total_words)

  total_sentences = tokenize.sent_tokenize(filtered)
  total_sent_len = len(total_sentences)
  
#Term Frequency Score
  tf_score = {}
  for each_word in total_words:
    each_word = each_word.replace('.','')
    if each_word not in stop_words:
      if each_word in tf_score:
          tf_score[each_word] += 1
      else:
          tf_score[each_word] = 1

# Dividing by total_word_length for each dictionary element
  tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
  def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

#Inverse Document Frequency Score
  idf_score = {}
  for each_word in total_words:
    each_word = each_word.replace('.','')
    if each_word not in stop_words:
      if each_word in idf_score:
          idf_score[each_word] = check_sent(each_word, total_sentences)
      else:
          idf_score[each_word] = 1

# Performing a log and divide
  idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

#print(idf_score)
  tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

#print(tf_idf_score)
  def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result

  st.write(get_top_n(tf_idf_score, 5))

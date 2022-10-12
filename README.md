# Darkset with Image Classification and Text Analysis 

To use this project follow below steps:

**1. Clone the github repository:**

!git clone https://github.com/akansha-singh7/Dataset/

**or**

git clone https://github.com/akansha-singh7/Dataset/

**2. Install Ngrok and Streamlit:**

!pip -q install streamlit

!pip -q install pyngrok

**or**

pip -q install streamlit

pip -q install pyngrok

**3. Import Streamlit and Ngrok:**

import pyngrok

import streamlit

**4. Get unique ngrok authentication key by logging in https://dashboard.ngrok.com/get-started/setup.**

**After getting the unique key run below command:**

!ngrok authtoken xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

**or**

ngrok authtoken xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

**5. Run below command to create public url:**
 
from pyngrok import ngrok

public_url = ngrok.connect(port='80')

print (public_url)

!streamlit run --server.port 80 /content/Dataset/app.py >/dev/null

**Note: Test and Test_HS folders contains test images and files**



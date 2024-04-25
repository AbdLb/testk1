
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/process"

page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover !important;
}
</style>
'''

st.write(page_bg_img, unsafe_allow_html=True)  # Utiliser st.write peut parfois donner un meilleur résultat


entity_name = st.text_input("Please enter the name of the entity:", "")

if st.button("Enter"):
    if entity_name:
        data = {"entity_name": entity_name}
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            summary = response.json()["summary"]
            st.write("Report:", summary)
        else:
            st.error(f"Error while summarizing. Status code: {response.status_code}")
    else:
        st.write("Please enter the name of the entity to have a KYC report")


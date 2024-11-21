import streamlit as st
import google.generativeai as genai
import PIL.Image
import os
import tempfile
import time
from dotenv import load_dotenv
from prompt import prompt

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


# Configure the Google API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# st.image("logo.png", width=200)

# Streamlit app title
st.title("Medical Prescription Reader")


model_choice = "gemini-1.5-Pro"


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image of the prescription...", type=["jpg", "jpeg", "png"]
)


# Generate Response button
if st.button("Analyze Prescription"):
    if uploaded_file is not None:
        # Display the uploaded image
        image = PIL.Image.open(uploaded_file)
        st.image(
            image, caption="Uploaded Prescription Image.", use_container_width=True
        )

        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tfile:
            tfile.write(uploaded_file.read())
            image_path = tfile.name

        # Upload the image to Gemini for analysis (OCR + explanation)
        st.write("Analyzing prescription...")

        # Use Gemini 1.5-Flash model (Model F)
        model = genai.GenerativeModel(model_name=f"models/{model_choice.lower()}")

        # Generate response with the model (sending image and prompt)
        response = model.generate_content(
            [prompt, image],
        )

        # Display the result
        st.write("Response:")
        st.write(response.text)

    else:
        st.write("Please upload an image of the prescription.")

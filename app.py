import streamlit as st
from mistralai import Mistral
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("MISTRAL_API_KEY is missing. Please add it to your .env file.")
    st.stop()

# Initialize Mistral client
client = Mistral(api_key=api_key)


# Function to send the image URL to Mistral and get analysis
def analyze_prescription(image_url):
    prompt = (
        "You are a medical assistant capable of analyzing images of handwritten doctor's prescriptions. "
        "The user has provided a URL to the prescription image. Perform the following tasks:\n"
        "1. Extract all handwritten text from the prescription (OCR).\n"
        "2. Identify all the drugs mentioned in the prescription.\n"
        "3. For each drug, provide general information, including its purpose, common uses, and any critical details "
        "a non-medical person should know.\n"
        "Provide the results in a structured and concise format."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url},
            ],
        }
    ]

    response = client.chat.complete(model="pixtral-12b-2409", messages=messages)
    return response.choices[0].message.content


# Streamlit UI
st.title("Doctor's Prescription Analyzer")
st.write(
    "Provide the URL of a doctor's handwritten prescription, and this app will use Mistral to extract the text (OCR) "
    "and analyze the drugs mentioned in it."
)

# Input URL for prescription image
image_url = st.text_input("Enter the URL of the prescription image:")

if image_url:
    try:
        # Display the image from the URL
        st.image(image_url, caption="Prescription Image", use_container_width=True)

        # Send the URL to Mistral for analysis
        st.write("Analyzing the prescription...")
        result = analyze_prescription(image_url)

        # Display the result
        st.success("Analysis Result:")
        st.write(result)
    except Exception as e:
        st.error(f"Error: {e}")

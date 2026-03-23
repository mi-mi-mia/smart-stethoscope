import streamlit as st
import requests

API_URL = "https://stetho-api-248800634185.europe-west1.run.app/predict"

st.set_page_config(page_title="Smart Stethoscope", page_icon="🫁")

st.title("Smart Stethoscope")
st.write("Upload a respiratory audio file and its annotation file to get a prediction.")

# Upload both required files
audio_file = st.file_uploader("Upload audio file (.wav)", type=["wav"])
annotation_file = st.file_uploader("Upload annotation file (.txt)", type=["txt"])

# Optional: play the uploaded audio in the UI
if audio_file is not None:
    st.audio(audio_file)

if st.button("Run prediction"):
    if audio_file is None or annotation_file is None:
        st.error("Please upload both the audio file and the annotation file.")
    else:
        files = {
            "audio_file": (audio_file.name, audio_file.getvalue(), "audio/wav"),
            "annotation_file": (annotation_file.name, annotation_file.getvalue(), "text/plain"),
        }

        try:
            with st.spinner("Sending files to API and running prediction..."):
                response = requests.post(API_URL, files=files, timeout=120)

            st.write(f"Status code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction complete")
                st.subheader("Result")
                st.write(f"**Prediction:** {result['prediction']}")
                st.write(f"**Cycles analysed:** {result['cycles_analysed']}")

                st.subheader("Cycle-level predictions")
                st.write(result["cycle_predictions"])

            else:
                st.error("The API returned an error.")
                st.text(response.text)

        except requests.exceptions.Timeout:
            st.error("The request timed out. The API may just be taking a while to process.")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

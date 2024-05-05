import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import logging
from google.cloud import logging as cloud_logging
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from datetime import (
    date,
    timedelta,
)
import whisper



# configure logging
logging.basicConfig(level=logging.INFO)
# attach a Cloud Logging handler to the root logger
log_client = cloud_logging.Client()
log_client.setup_logging()

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    text_model_pro = GenerativeModel("gemini-1.0-pro")
    multimodal_model_pro = GenerativeModel("gemini-1.0-pro-vision")
    whisper_model = whisper.load_model("base")
    return text_model_pro, multimodal_model_pro, whisper_model


def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
    responses = model.generate_content(
        prompt_list, generation_config=generation_config, stream=stream
    )
    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            pass
    return "".join(final_response)

st.header("Gemini 1.0 Pro Vision - AI Doctor", divider="gray")
text_model_pro, multimodal_model_pro, whisper_model = load_models()


st.write("Audio to Arztbrief translator")
st.subheader("AI Doc")


st.text('Patientenaufnahme')
st.text('Bitte erwähnen: Vorname, Name, Geburtsdatum, Aufnamedatum des Patienten, Diagnosen, Behandlung')
audio_bytes = audio_recorder(
    text="",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="user",
    icon_size="6x",
)
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with open("audio.wav", "wb") as binary_file:
        binary_file.write(audio_bytes)

    audio_file = Part.from_data(audio_bytes, mime_type="audio/wav")


cuisine = st.selectbox(
    "What field of medicine are you referring to?",
    ("Internal", "Family", "Pediatrics", "Dermatology"),
    index=None,
    placeholder="Select your desired discipline."
)

max_output_tokens = 2048

prompt = """
Only answer with information in the attached audio file. Transcribe the audio file, then answer the following questions: \n
        - What is the name of the person? \n
        - What is the date of birth (this is the first date mentioned)? \n
        - When was the patient first submitted (this is the second date mentioned)? \n
        - What was the medical diagnose on the patients?
        - Which steps were taken?

Please give additional information on the audio file:
        - how long is it?
        - what language is it in?
        - how is the audio quality?

If there is no audio file, please say so.
"""

config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
}

generate_t2t = st.button("Generate doctor's letter...", key="generate_t2t")
if generate_t2t and prompt:
    with st.spinner("Generating your doctor's letter using Gemini..."):
        first_tab1, first_tab2 = st.tabs(["Result", "Prompt"])
        with first_tab1:
            response = get_gemini_pro_vision_response(
                multimodal_model_pro,
                [audio_file, prompt],
                generation_config=config,
            )
            if response:
                st.write("Your doctor's letter:")
                st.write(response)
                logging.info(response)
        with first_tab2:
            st.text(prompt)

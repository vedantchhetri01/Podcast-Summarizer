import gradio as gr
from transformers import pipeline
import whisper
import yt_dlp as youtube_dl
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_audio_from_youtube(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',  # Best audio quality
            'postprocessors': [],  # No post-processing (skip audio conversion)
            'outtmpl': 'downloads/%(id)s.%(ext)s',  # Save to 'downloads' folder
            'quiet': True,  # Suppress unnecessary output
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_file = f"downloads/{info_dict['id']}.webm" 
            return audio_file
    except Exception as e:
        return f"Error while downloading audio: {str(e)}"

def transcribe_and_summarize(url):
    if "youtube.com" in url or "youtu.be" in url:
        audio_file = extract_audio_from_youtube(url)
    else:
        return "Unsupported URL format. Please provide a valid YouTube URL.", ""
    if isinstance(audio_file, str) and "Error" in audio_file:
        return audio_file, ""

    transcription = whisper_model.transcribe(audio_file)
    text = transcription['text']

    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    
    return text, summary[0]['summary_text']

css = """
    body {
        background: linear-gradient(to bottom right, #FEFOED, #ACE3F8);  /* Gradient background */
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 0;
        color: #fff;
    }
    .gradio-container {
        background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent white container */
        border-radius: 15px;
        padding: 40px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        max-width: 900px;
        margin: auto;
        backdrop-filter: blur(10px);  /* Blurred background effect */
    }
    .gradio-header {
        color: #fff;
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .gradio-description {
        text-align: center;
        font-size: 18px;
        color: #ddd;
        margin-bottom: 40px;
    }
    .gradio-button {
        background-color: #2575fc;
        color: white;
        font-size: 18px;
        padding: 12px 30px;
        border: none;
        border-radius: 8px;
        transition: background-color 0.3s, transform 0.3s ease-in-out;
        cursor: pointer;
    }
    .gradio-button:hover {
        background-color: #1e63cc;
        transform: scale(1.05);
    }
    .gradio-input {
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        border: 1px solid #ccc;
        margin-bottom: 30px;
        transition: border 0.3s ease;
    }
    .gradio-input:focus {
        border: 1px solid #2575fc;
        outline: none;
    }
    .gradio-textbox {
        font-size: 16px;
        border-radius: 8px;
        padding: 12px;
        background-color: #f1f1f1;
        border: 1px solid #ccc;
        color: #333;
        margin-bottom: 20px;
    }
    .gradio-textbox[readonly] {
        background-color: #fafafa;
    }
    .gradio-footer {
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        color: #ddd;
    }
"""
interface = gr.Interface(
    fn=transcribe_and_summarize,
    inputs=gr.Textbox(label="Enter YouTube URL", placeholder="Paste YouTube URL here...", lines=1),
    outputs=[gr.Textbox(label="Full Transcript", interactive=False), gr.Textbox(label="Summary", interactive=False)],
    title="Podcast Summarizer from YouTube🧾",
    description="Paste a YouTube URL to get the transcription and summary of the podcast episode. The text will be processed and summarized in real-time.",
    css=css
)
interface.launch(share=True)

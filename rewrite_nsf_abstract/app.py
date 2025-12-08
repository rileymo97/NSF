import os
import json
import gradio as gr
from google import genai

# Get banned words 
BANNED_WORDS_RAW = os.environ.get("BANNED_WORDS", "[]")
try:
    BANNED_WORDS = json.loads(BANNED_WORDS_RAW)
except json.JSONDecodeError:
    BANNED_WORDS = []

# Configure Gemini with the new SDK
GEMINI_KEY = os.environ.get("API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

# Rewrite function
def rewrite_abstract(abstract):
    if not abstract or not abstract.strip():
        return "Please enter an abstract to rewrite."
    
    prompt = (
        "Rewrite the following NSF-style abstract with these rules:\n"
        "1. ONLY rewrite sentences that contain these banned words or their variations: "
        f"{', '.join(BANNED_WORDS)}.\n"
        "2. For sentences WITHOUT banned words, keep them EXACTLY as written.\n"
        "3. For sentences WITH banned words, replace the banned words with neutral, "
        "scientific alternatives while keeping all factual meaning intact.\n"
        "4. Maintain the same structure, tone, and style of the original abstract.\n\n"
        f"Original abstract:\n{abstract}\n\n"
        "Rewritten abstract:"
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# Interface
iface = gr.Interface(
    fn=rewrite_abstract,
    inputs=gr.Textbox(lines=10, label="Enter the abstract to rewrite:"),
    outputs=gr.Textbox(lines=10, label="Rewritten Abstract:"),
    title="Abstract Rewriter",
    description="Enter an abstract and get a rewritten version.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
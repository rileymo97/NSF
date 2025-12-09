# NSF Abstract Rewriter

A Gradio web application that rewrites NSF grant abstracts by replacing banned words with neutral, scientific alternatives while preserving all factual content and maintaining the original structure and style.

Currently hosted at https://huggingface.co/spaces/evanfantozzi/rewrite-nsf-abstract 

## Setup

1. Install dependencies:
```bash
uv run pip install -r requirements.txt
```

2. Set environment variables:
```bash
export API_KEY="your-google-gemini-api-key" # Replace with your actual API key
export BANNED_WORDS='[
    "activism",
    "activists",
    "advocacy",
    "advocate",
    "barrier",
    "barriers",
    "biased",
    "bias",
    "BIPOC",
    "Black and Latinx",
    "community diversity",
    "community equity",
    "cultural differences",
    "cultural heritage",
    "culturally responsive",
    "disabilities",
    "discrimination",
    "discriminatory",
    "backgrounds",
    "groups",
    "diversified",
    "diversify",
    "enhancing",
    "equal opportunity",
    "equality",
    "equitable",
    "ethnicity",
    "excluded",
    "female",
    "fostering",
    "gender",
    "hate speech",
    "Hispanic minority",
    "historically",
    "implicit bias",
    "inclusion",
    "inclusive",
    "increase",
    "indigenous community",
    "inequalities",
    "inequities",
    "institutional",
    "LGBTQ",
    "marginalize",
    "minorities",
    "multicultural",
    "polarization",
    "political",
    "prejudice",
    "privileges",
    "promoting",
    "race",
    "racial",
    "justice",
    "sense of belonging",
    "sexual preferences",
    "social justice",
    "sociocultural",
    "socioeconomic",
    "status",
    "stereotypes",
    "systemic",
    "trauma",
    "underappreciated",
    "underrepresented",
    "underserved",
    "victim",
    "women"
]''
```

3. Run the app:
```bash
uv run python app.py
```

## Usage

1. Enter an NSF-style abstract in the input textbox
2. Click submit to get the rewritten version
3. The app will:
   - Keep sentences without banned words exactly as written
   - Rewrite sentences with banned words using neutral alternatives
   - Maintain the same structure, tone, and style


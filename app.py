from flask import Flask, request, render_template_string
from PIL import Image
import requests
import io
import base64

app = Flask(__name__)

# Hugging Face public models (no API key required)
VISION_MODEL = "Salesforce/blip-image-captioning-base"
LLM_MODEL = "google/flan-t5-small"

VISION_URL = f"https://api-inference.huggingface.co/models/{VISION_MODEL}"
LLM_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"

HTML = """
<!doctype html>
<title>Image Summary AI</title>
<h2>Upload an image</h2>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*" required>
  <input type="submit" value="Analyze">
</form>

{% if caption %}
<h3>Image description</h3>
<p>{{ caption }}</p>

<h3>Summary</h3>
<p><b>{{ summary }}</b></p>
{% endif %}
"""

def image_to_caption(img_bytes):
    r = requests.post(VISION_URL, data=img_bytes)
    r.raise_for_status()
    data = r.json()
    return data[0]["generated_text"]

def summarize(text):
    payload = {
        "inputs": f"Summarize this image description in one sentence: {text}"
    }
    r = requests.post(LLM_URL, json=payload)
    r.raise_for_status()
    data = r.json()
    return data[0]["generated_text"]

@app.route("/", methods=["GET", "POST"])
def upload():
    caption = None
    summary = None

    if request.method == "POST":
        file = request.files["image"]
        img_bytes = file.read()

        caption = image_to_caption(img_bytes)
        summary = summarize(caption)

    return render_template_string(HTML, caption=caption, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)

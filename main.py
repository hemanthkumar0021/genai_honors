import os
import openai
import json
from config import OPENAI_API_KEY, API_PROVIDER
from datetime import datetime

# --- SETUP API ---
openai.api_key = OPENAI_API_KEY

# --- Sentiment/Emotion Detection ---
def detect_emotion(review):
    """Uses GPT-style model to detect emotion and key points from text."""
    prompt = f"""
    Analyze the following customer review. 
    1. Provide the primary emotion (one word).
    2. Summarize the key points in short bullet points.

    Review: "{review}"
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message["content"]

# --- Image Prompt Builder ---
def build_image_prompt(emotion_analysis):
    """Turn emotion + key points into a creative image prompt."""
    return f"An artistic illustration representing: {emotion_analysis}. High detail, vibrant colors."

# --- Image Generation ---
def generate_image(prompt, filename):
    """Generate an image using DALL·E API."""
    image_resp = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = image_resp["data"][0]["url"]

    # Download and save image
    import requests
    img_data = requests.get(image_url).content
    with open(filename, "wb") as handler:
        handler.write(img_data)
    print(f"Image saved as {filename}")

# --- Main Workflow ---
def main():
    os.makedirs("outputs", exist_ok=True)
    with open("reviews.txt", "r", encoding="utf-8") as f:
        reviews = [line.strip() for line in f if line.strip()]

    for i, review in enumerate(reviews, start=1):
        print(f"\nProcessing review {i}/{len(reviews)}...")
        
        # Step 1: Get emotion + key points
        emotion_analysis = detect_emotion(review)
        print("Emotion Analysis:", emotion_analysis)

        # Step 2: Build AI image prompt
        image_prompt = build_image_prompt(emotion_analysis)
        print("Image Prompt:", image_prompt)

        # Step 3: Generate and save image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        img_filename = f"outputs/review_{i}_{timestamp}.png"
        generate_image(image_prompt, img_filename)

if __name__ == "__main__":
    main()

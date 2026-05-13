# Code adapted from: https://www.gradio.app/docs/gradio/interface
# https://www.gradio.app/docs/gradio/textbox
# https://pyabsa.readthedocs.io/en/latest/1_quick_start/atesc.html

from pyabsa import AspectTermExtraction as ATEPC
import gradio as gr
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json

load_dotenv()

aspect_extractor = ATEPC.AspectExtractor('English')

# Global storage for review snippets (organized by aspect)
# This will be updated each time analyze is clicked.
aspect_reviews_storage = {}
global_aspect_counts = {}

def sentiment_classifier(text):
    global aspect_reviews_storage, global_aspect_counts
    # Reset storage for new batch
    aspect_reviews_storage = {}
    global_aspect_counts = {}
    
    # Split by double newlines so reviews can contain their own line breaks
    reviews = text.strip().split("\n\n")
    
    aspect_instance = {}
    all_results = []

    for review in reviews:
        review_text = review.strip()
        if not review_text: continue
            
        review_lower = review_text.lower()
        result = aspect_extractor.predict(review_lower)
        all_results.append(result)
        aspects = result.get("aspect", [])
        sentiments = result.get("sentiment", [])

        for i in range(len(aspects)):
            aspect = aspects[i]
            sentiment = sentiments[i]
            
            if aspect not in aspect_instance:
                aspect_instance[aspect] = {"Aspect" : aspect,"Positive": 0, "Negative": 0, "Neutral": 0, "Total": 0 }
                aspect_reviews_storage[aspect] = []
                
            aspect_instance[aspect]["Total"] += 1
            aspect_reviews_storage[aspect].append(review_text)

            if sentiment == "Positive":
                aspect_instance[aspect]["Positive"] += 1
            elif sentiment == "Negative":
                aspect_instance[aspect]["Negative"] += 1
            else: 
                aspect_instance[aspect]["Neutral"] += 1
    with open("all_review_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    if not aspect_instance:
        print("No aspects found in the input text.")
        return pd.DataFrame(columns=["Aspect", "Positive", "Negative", "Neutral", "Total"]), gr.update(choices=[], value=None)
    # Setting up table so that the columns are in the correct order and sorted alphabetically
    global_aspect_counts = aspect_instance
    aspect_choices = sorted(aspect_instance.keys(), key=str.lower)
    table = pd.DataFrame.from_dict(aspect_instance, orient="index")
    table = table[["Aspect", "Positive", "Negative", "Neutral", "Total"]]
    table = table.loc[aspect_choices]
    
    aspect_choices = ["All Aspects"] + aspect_choices
    print(f"Found aspects: {aspect_choices}")
    return gr.update(value=table), gr.update(choices=aspect_choices, value=aspect_choices[0] if aspect_choices else None)

def summarize_aspect(aspect):
    token = os.getenv("HF_TOKEN")
    if not token:
        return "Error: No Hugging Face token found. Please set the HF_TOKEN environment variable"
    if not aspect or (aspect != "All Aspects" and aspect not in aspect_reviews_storage):
        return "No reviews found for this aspect."

    client = InferenceClient(token=token)
    
    if aspect == "All Aspects":
        # Combine snippets from all aspects
        # Limit can be adjusted if needed.
        all_snippets = []
        for a in aspect_reviews_storage:
            all_snippets.extend(aspect_reviews_storage[a][:3])
        reviews_text = "\n- ".join(all_snippets[:15])
        target_desc = "all features and the overall product"
    else:
        reviews_text = "\n- ".join(aspect_reviews_storage[aspect][:10]) 
        target_desc = f"the '{aspect}'"

    # Current prompt to test generic aspect summaries.
    # Can be adjusted in the future if we're not happy with the results.
    prompt = (
        f"You are a sentiment analyst. Based only on the following reviews, "
        f"summarize what people say about {target_desc}.\n"
        f"Do not mention any other product features or aspects besides {target_desc}.\n"
        f"Reviews:\n- {reviews_text}\n\n"
        f"Format your response as follows:\n"
        f"Summary: [Your 2-sentence summary]"
    )
    
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_tokens=200
        )
        result_text = response.choices[0].message.content

        result_text = result_text.replace("Summary:", "").strip()

        return result_text

    except Exception as e:
        return f"Error calling LLM: {str(e)}"


with gr.Blocks(title="Aspect Based Sentiment Review Tool") as demo:
    gr.Markdown("# Aspect Based Sentiment Review Tool")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=10, label="Paste Reviews (separated by double newlines)", placeholder="Review 1: The battery is great...\n\nReview 2: I hate the camera...")
            analyze_btn = gr.Button("Analyze Sentiments", variant="primary")
        
        with gr.Column():
            output_table = gr.Dataframe(
                label="Overall Summary",
                headers=["Aspect", "Positive", "Negative", "Neutral", "Total"],
                datatype=["str", "number", "number", "number", "number"]
            )

    with gr.Row():
        with gr.Column():
            aspect_dropdown = gr.Dropdown(label="Select Aspect to Summarize", choices=[])
            summarize_btn = gr.Button("Generate AI Summary")
        
        with gr.Column():
            summary_output = gr.Textbox(label="Llama 3.1 Summary", lines=4)

    analyze_btn.click(
        fn=sentiment_classifier,
        inputs=[input_text],
        outputs=[output_table, aspect_dropdown]
    )
    
    summarize_btn.click(
        fn=summarize_aspect,
        inputs=[aspect_dropdown],
        outputs=[summary_output]
    )

if __name__ == "__main__":
    demo.launch()

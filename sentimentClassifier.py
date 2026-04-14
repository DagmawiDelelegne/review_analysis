# Code adapted from: https://www.gradio.app/docs/gradio/interface
# https://www.gradio.app/docs/gradio/textbox
# https://pyabsa.readthedocs.io/en/latest/1_quick_start/atesc.html

from pyabsa import AspectTermExtraction as ATEPC
import gradio as gr
import pandas as pd

aspect_extractor = ATEPC.AspectExtractor('multilingual')

def sentiment_classifier(text):
    reviews = text.strip().split("\n")
    
    aspect_instance = {}

    for review in reviews:
        review = review.lower()
        result = aspect_extractor.predict(review)
        aspects = result.get("aspect")
        sentiments = result.get("sentiment")

        for i in range (len(aspects)):
            
            aspect = aspects[i]
            sentiment = sentiments[i]
            if aspect not in aspect_instance:
                aspect_instance[aspect] = {"Aspect" : aspect,"Positive": 0, "Negative": 0, "Neutral": 0, "Total": 0 }
                
            aspect_instance[aspect]["Total"] += 1

            if (sentiment == "Positive"):
                aspect_instance[aspect]["Positive"] += 1
            elif (sentiment == "Negative"):
                aspect_instance[aspect]["Negative"] += 1
            else: 
                aspect_instance[aspect]["Neutral"] += 1
    
    table = pd.DataFrame.from_dict(aspect_instance,orient="index")
    return table


interface = gr.Interface(sentiment_classifier, inputs=gr.Textbox(lines = 12), outputs = gr.Dataframe(), title="Aspect Based Sentiment Review Tool")     
            

interface.launch()

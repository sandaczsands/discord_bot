from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if len(text.split()) < 50:
        return "Zbyt mało treści do podsumowania."
    result = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return result[0]['summary_text']

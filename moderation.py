from transformers import pipeline

moderation_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

def is_inappropriate(text):
    result = moderation_classifier(text)[0]
    return result['label'].lower() == 'spam'

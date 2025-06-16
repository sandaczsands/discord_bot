def is_inappropriate(text, pipeline_model):
    result = pipeline_model(text)[0]
    return result['label'].lower() == 'spam'

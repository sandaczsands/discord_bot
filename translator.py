from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, target_lang):
    if target_lang == "en":
        model_name = "Helsinki-NLP/opus-mt-pl-en"
    elif target_lang == "pl":
        model_name = "Helsinki-NLP/opus-mt-en-pl"
    else:
        return "Nieobsługiwany język docelowy."

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

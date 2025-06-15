from transformers import T5Tokenizer, T5ForConditionalGeneration

def translate_text(command):
    # Remove command prefix and split parts
    parts = command.strip().split(" ", 3)

    _, source_lang, target_lang, text = parts

    # Define language mapping for T5
    lang_map = {
        "en": "English",
        "pl": "Polish",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
    }

    if source_lang not in lang_map or target_lang not in lang_map:
        return "Unsupported language code. Use: en, pl, de, fr, es"

    # Build the T5 input
    task_prefix = f"translate {lang_map[source_lang]} to {lang_map[target_lang]}: "
    input_text = task_prefix + text

    # Load model and tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and translate
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128)
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return translation

def translate_text(command, model, tokenizer):
    parts = command.strip().split(" ", 3)
    _, source_lang, target_lang, text = parts

    lang_map = {
        "en": "English",
        "pl": "Polish",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
    }

    if source_lang not in lang_map or target_lang not in lang_map:
        return "Unsupported language code. Use: en, pl, de, fr, es"

    task_prefix = f"translate {lang_map[source_lang]} to {lang_map[target_lang]}: "
    input_text = task_prefix + text

    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=128)
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return translation

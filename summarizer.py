def summarize_text(text, model, tokenizer):
    if len(text.split()) < 50:
        return "Zbyt mało treści do podsumowania."
    
    input_text = f"summarize: {text}"
    print(f"Input for summarization: {input_text}")  
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=5)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary
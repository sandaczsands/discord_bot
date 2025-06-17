def answer_question(question, context, model, tokenizer):
    if len(context.split()) < 50:
        return "Zbyt mało kontekstu, by odpowiedzieć."

    input_text = f"question: {question} context: {context}"
    print(f"Input for question answering: {input_text[:100]}") 
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=100)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

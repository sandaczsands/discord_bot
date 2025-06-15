from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_question(question, context):
    if len(context.split()) < 50:
        return "Zbyt mało kontekstu, by odpowiedzieć."
    result = qa_pipeline(question=question, context=context)
    return result['answer']

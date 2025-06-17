from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MessageModerator:
    def __init__(self, spam_threshold=0.5, similarity_threshold=0.5):
        self.spam_threshold = spam_threshold
        self.similarity_threshold = similarity_threshold
        self.pipeline_model = pipeline(
            "text-classification",
            model="cointegrated/rubert-tiny-toxicity",
            tokenizer="cointegrated/rubert-tiny-toxicity"
        )
        self.label_map = {
            "insult": "obelga",
            "obscenity": "wulgaryzm",
            "threat": "groźba",
            "dangerous": "niebezpieczna treść",
            "non-toxic": None
        }

    async def is_repeated(self, message_obj):
        async for msg in message_obj.channel.history(limit=10):
            if msg.id == message_obj.id:
                continue
            if msg.author == message_obj.author and msg.content == message_obj.content:
                print(f"Repeated message detected: {msg.content}")
                return True
        return False

    def compute_similarity(self, text1, text2):
        texts = [text1, text2]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]

    async def is_too_similar(self, message_obj):
        async for msg in message_obj.channel.history(limit=10):
            if msg.author == message_obj.author:
                if msg.id == message_obj.id:
                    continue
                similarity = self.compute_similarity(message_obj.content, msg.content)
                if similarity > 1 - self.similarity_threshold:
                    print(f"Too similar message detected: {msg.content}")
                    return True
        return False

    def is_spam(self, message_obj):
        results = self.pipeline_model(message_obj.content)
        print(f"Toxicity detection results: {results}")

        for result in results:
            if result['score'] >= self.spam_threshold and result['label'] != 'non-toxic':
                label = result['label']
                reason = self.label_map.get(label, "toksyczna wiadomość")
                print(f"Detected toxic content ({label}): {message_obj.content}")
                return True, reason
        return False, None

    async def is_inappropriate(self, message_obj):
        is_spam, spam_reason = self.is_spam(message_obj)
        if is_spam:
            return True, spam_reason

        if await self.is_repeated(message_obj):
            return True, "powtarzająca się wiadomość"

        if await self.is_too_similar(message_obj):
            return True, "zbyt podobna wiadomość"

        return False, None

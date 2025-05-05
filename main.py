import torch
from sentence_transformer import SentenceTransformer

def demo_sentence_transformer():
    model = SentenceTransformer()

    # My two input sentences
    sentences = [
        "My favorite coffee shop just reopened downtown.",
        "The weather has been unusually warm this week."
    ]
    
    # Turns sentences into embeddings
    embeddings = model.encode(sentences)

    print("\nSentence Transformer Demo:")
    for sentence, embedding in zip(sentences, embeddings):
        print(f"Sentence: {sentence}")
        print(f"Embedding: {embedding[:5]}...")  # Print only first 5 values for brevity

if __name__ == "__main__":
    demo_sentence_transformer()

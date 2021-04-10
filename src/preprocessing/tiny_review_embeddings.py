from transformers import AutoTokenizer, AutoModel
from nltk import tokenize as nltk_tokenize
import numpy as np
import torch
import nltk

# Downloads
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

# Settings
sentence_embedding_type = "CLS"  # "avg"
review_embedding_type = "avg"  # other?
useGPU = True if torch.cuda.is_available() else False
    
if useGPU:
    model.to("cuda")

def process(sentence):
    """Pre-process sentence(s) for BERT. Returns:
        - tokenized text (with [CLS] and [SEP] tokens)
        - segment sentence ids ([0s & 1s])
        - indexed tokens """
    tokenized_text = ["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"]
    tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(tokenized_text))]
    return tokenized_text, segments_ids, indexed_tokens


def get_sentence_embedding(sentence):
    """returns sentence (pair) embedding of size [1, 128]"""
    tokenized_review, segments_ids, indexed_tokens = process(sentence)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    if useGPU:
        tokens_tensor = tokens_tensor.to("cuda")
        segments_tensors = segments_tensors.to("cuda")

    # get token embeddings
    with torch.no_grad():
        output = model(tokens_tensor, token_type_ids=segments_tensors)

    # get pooled embedding using specified method
    if sentence_embedding_type == "CLS":
        return output.pooler_output
    elif sentence_embedding_type == "avg":
        return output.last_hidden_state.mean(axis=1)
    else:
        return None


def get_review_embedding(review):
    """returns review embedding of size [1, 128]"""
    review_sentences = nltk_tokenize.sent_tokenize(review)
    sentence_embeddings = list(map(get_sentence_embedding, review_sentences))
    if len(sentence_embeddings) == 0:
        print("Sentence_embeddings are empty!")
        print(review)
        return torch.zeros(1,128)
    if review_embedding_type == "avg":
        # avg over all pairs [pairs, 1, 128] => [1, 128]
        mean = torch.mean(torch.stack(sentence_embeddings), axis=0)
        return mean


def get_embedding(reviews):
    embeddings = []
    for review in reviews:
        # [1, 128]
        embeddings.append(get_review_embedding(review))
    # [num_reviews, 1, 128] => [1, 128]
    mean = torch.mean(torch.stack(list(embeddings)), axis=0)
    return mean


if __name__ == "__main__":
    example_review = "I love this product. \
            I mean honestly, who doesn't love chocolate? \
            Only sociopaths, I reckon. \
            I'd eat this every day if I could. "

    example_review2 = "I hate this product. \
            Wish I could ban it from existing "

    example_review3 = "Five Stars."

    with open("src/preprocessing/long_example.txt") as f:
        example_review4 = f.read()

    example_reviews = [example_review, example_review2, example_review3, example_review4]

    get_embedding(example_reviews)

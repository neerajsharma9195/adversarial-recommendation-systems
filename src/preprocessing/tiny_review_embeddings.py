from transformers import AutoTokenizer, AutoModel
from nltk import tokenize as nltk_tokenize
import numpy as np
import torch
import nltk

# Settings
sentence_embedding_type = "CLS"  # "avg"
review_embedding_type = "avg"  # other?
    

def process(first_sentence, second_sentence):
    """Pre-process sentence(s) for BERT. Returns:
        - tokenized text (with [CLS] and [SEP] tokens)
        - segment sentence ids ([0s & 1s])
        - indexed tokens """
    # Downloads
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    tokenized_first = ["[CLS]"] + tokenizer.tokenize(first_sentence) + ["[SEP]"]
    if second_sentence:
        tokenized_second = tokenizer.tokenize(second_sentence) + ["[SEP]"]
    else:
        tokenized_second = []
    tokenized_text = tokenized_first + tokenized_second
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(tokenized_first))]
    segments_ids += [1 for i in range(len(tokenized_second))]
    return tokenized_text, segments_ids, indexed_tokens


def get_sentence_embedding(sentence_pair):
    """returns sentence (pair) embedding of size [1, 768]"""
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()

    first_sentence, second_sentence = sentence_pair
    tokenized_review, segments_ids, indexed_tokens = process(
        first_sentence, second_sentence
    )
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    if torch.cuda.is_available():
        model.to("cuda")
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


def create_sentence_pairs(review):
    """ returns array of sentence pair tuples"""
    pairs = []
    review_sentences = nltk_tokenize.sent_tokenize(review)
    first_sentence, last_sentence = review_sentences[0], review_sentences[-1]

    # first sentence alone
    pairs.append((first_sentence, ""))

    # all pairs of sentences
    for line in review_sentences[1:]:
        second_sentence = line
        pairs.append((first_sentence, second_sentence))
        first_sentence = line

    # last sentence alone
    if len(review_sentences) > 1:
        pairs.append((last_sentence, ""))
    return pairs


def get_review_embedding(review):
    """returns review embedding of size [1, 768]"""
    sentence_pairs = create_sentence_pairs(review)
    sentence_embeddings = map(get_sentence_embedding, sentence_pairs)  # [pairs, 1, 768]
    if review_embedding_type == "avg":
        # avg over all pairs [pairs, 1, 768] => [1, 768]
        mean = torch.mean(torch.stack(list(sentence_embeddings)), axis=0)
        return mean


def get_embedding(reviews):
    embeddings = []
    for review in reviews:
        # [1, 768]
        embeddings.append(get_review_embedding(review))
    # [num_reviews, 1, 768] => [1, 768]
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

    example_reviews = [example_review, example_review2, example_review3]

    get_embedding(example_reviews)

import json
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



with open('configs/vocab.json', 'r') as f:
    vocab = json.load(f)


def tokenize(sentence):
    words_ending_with_dot = ''.join([letter for letter in sentence.lower() if letter != '.'])
    tokens_ = [words for words in  words_ending_with_dot.split()]

    #remove numbers
    without_numbers = []
    for word in tokens_:
        if re.search(r'\d', word):
            pass
        else:
            without_numbers.append(word)


    # remove
    without_special_characters = []

    for word in without_numbers:
        if re.search(r'[^\w\s]', word):
            pass
        else:
            without_special_characters.append(word)

    # remove stopwords

    without_stopwords = []

    for word in without_special_characters:
        if word in stop_words:
            continue
        else:
            without_stopwords.append(word)

    return without_stopwords


def create_idx(tokens: list):
    ind = []
    for word in tokens:
        idx = vocab.get(word,vocab['<UNK>'])
        ind.append(idx)
    return ind

        

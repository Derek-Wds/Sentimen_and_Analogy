import string, re, math
from stop_list import *
from nltk.corpus import stopwords

# Load NLTK's English stop-words list and stop list provided by Professor Adam Meyers at NYU
closed_class_stop_words.extend(stopwords.words('english'))
stop_words = set(closed_class_stop_words)


# function to split the string to tokens and words, and remove those are stop words
def split_sentence(sentence):
    tokens = sentence.split()
    result = list()
    for token in tokens:
        if token in stop_words or token in string.punctuation:
            pass
        else:
            if re.search('[a-zA-Z]', token):
                result.append(token)
    return result


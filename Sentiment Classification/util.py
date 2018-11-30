import string, re, math

# function to split the string to tokens and words, and remove those are stop words
def split_sentence(sentence):
    tokens = sentence.split()
    result = list()
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            result.append(token)
    return result


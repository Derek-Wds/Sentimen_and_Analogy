import json, random, copy, random, re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
random.seed(3)


def read_files():
    data = dict()
    # data["noun-plural_reg"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I01 [noun - plural_reg].txt"))
    # data["noun+plural_irreg"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I02 [noun - plural_irreg].txt"))
    # data["adj-comparative"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I03 [adj - comparative].txt"))
    # data["adj-superlative"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I04 [adj - superlative].txt"))
    # data["verb_inf-3pSg"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I05 [verb_inf - 3pSg].txt"))
    # data["verb_inf-ving"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I06 [verb_inf - Ving].txt"))
    # data["verb_inf-ved"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I07 [verb_inf - Ved].txt"))
    # data["verb_ving-3psg"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I08 [verb_Ving - 3pSg].txt"))
    # data["verb_ving-ved"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I09 [verb_Ving - Ved].txt"))
    # data["verb_3psg-ved"] = get_pairs(read_file("BATS_3.0/1_Inflectional_morphology/I10 [verb_3pSg - Ved].txt"))
    # data["noun+less_reg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D01 [noun+less_reg].txt"))
    # data["un+adj_reg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D02 [un+adj_reg].txt"))
    # data["adj+ly_reg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D03 [adj+ly_reg].txt"))
    # data["over+adj_reg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D04 [over+adj_reg].txt"))
    # data["adj+ness_reg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D05 [adj+ness_reg].txt"))
    # data["re+verb_reg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D06 [re+verb_reg].txt"))
    # data["ver+able_reg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D07 [verb+able_reg].txt"))
    # data["verb+er_irreg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D08 [verb+er_irreg].txt"))
    # data["verb+tion_irreg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D09 [verb+tion_irreg].txt"))
    # data["verb+ment_irreg"] = get_pairs(read_file("BATS_3.0/2_Derivational_morphology/D10 [verb+ment_irreg].txt"))
    # data["country-capital"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E01 [country - capital].txt"))
    # data["country-language"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E02 [country - language].txt"))
    # data["UK_city+county"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E03 [UK_city - county].txt"))
    # data["name-nationality"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E04 [name - nationality].txt"))
    # data["name-occupation"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E05 [name - occupation].txt"))
    # data["animal-young"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E06 [animal - young].txt"))
    # data["animal-sound"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E07 [animal - sound].txt"))
    # data["animal-shelter"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E08 [animal - shelter].txt"))
    # data["things-color"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E09 [things - color].txt"))
    # data["male-female"] = get_pairs(read_file("BATS_3.0/3_Encyclopedic_semantics/E10 [male - female].txt"))
    data["hypernyms-animals"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L01 [hypernyms - animals].txt"))
    data["hypernyms-misc"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L02 [hypernyms - misc].txt"))
    data["hyponyms-misc"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L03 [hyponyms - misc].txt"))
    data["meronyms-substance"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L04 [meronyms - substance].txt"))
    data["meronyms-member"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L05 [meronyms - member].txt"))
    data["meronyms-part"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L06 [meronyms - part].txt"))
    data["synonyms-intensity"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L07 [synonyms - intensity].txt"))
    data["synomyms-exact"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L08 [synonyms - exact].txt"))
    data["antonyms-gradable"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L09 [antonyms - gradable].txt"))
    data["antonyms-binary"] = get_pairs(read_file("BATS_3.0/4_Lexicographic_semantics/L10 [antonyms - binary].txt"))

    return data


def get_pairs(d):
    length = len(d)
    data = dict()
    num = 0
    for i in range(length - 1):
        for j in range(i, length):
            Now = copy.deepcopy(d[i])
            Next = copy.deepcopy(d[j])
            Now.extend(Next)
            data[num] = Now
            num += 1
    return data

def read_file(file_path):
    data = dict()
    with open(file_path, "r") as f:
        lines = f.readlines()
        num = 0
        for line in lines:
            words = line.strip().split()
            if "/" in words[1]:
                additions = words[1].split("/")
                data[num] = [words[0], additions[0]]
                num += 1
                for word in additions[1:]:
                    a = random.randint(1, 10)
                    if a % 3 == 0:
                        data[num] = [words[0], word]
                        num += 1
            else:
                data[num] = [words[0], words[1]]
                num += 1
    return data

def plot(word2vec, fasttext, glove):
    labels = ["noun-plural_reg","noun+plural_irreg","adj-comparative","adj-superlative","verb_inf-3pSg","verb_inf-ving","verb_inf-ved","verb_ving-3psg","verb_ving-ved","verb_3psg-ved",\
            "noun+less_reg","un+adj_reg","adj+ly_reg","over+adj_reg", "adj+ness_reg","re+verb_reg","ver+able_reg","verb+er_irreg","verb+tion_irreg","verb+ment_irreg",\
            "country-capital","country-language","UK_city+county","name-nationality","name-occupation","animal-young","animal-sound","animal-shelter","things-color","male-female",\
            "hypernyms-animals","hypernyms-misc","hyponyms-misc","meronyms-substance","meronyms-member","meronyms-part","synonyms-intensity","synomyms-exact","antonyms-gradable","antonyms-binary"]

    fig, ax = plt.subplots(figsize=(10, 4))

    idx = np.asarray([i for i in range(len(labels))])

    width = 0.2

    ax.bar(idx, [val for key, val in sorted(word2vec.items())], width=width, color="red", alpha=0.6)
    ax.bar(idx+width, [val for key,
                       val in sorted(fasttext.items())], width=width, color="darkorange", alpha=0.6)
    ax.bar(idx+width*2, [val for key,
                         val in sorted(glove.items())], width=width, color="royalblue", alpha=0.6)
    ax.set_xticks(idx)
    ax.set_xticklabels(sorted(labels), rotation="vertical")
    ax.legend(['Word2vec', 'FastText', 'Glove'])
    ax.set_xlabel('Types')
    ax.set_ylabel('Accuracy')

    fig.tight_layout()

    plt.show()


def read_results(file_path):
    data = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        sentences = list()
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                sentences.append(line)
        length = len(sentences)
        for i in range(length):
            if i % 2 == 0:
                data[sentences[i]] = 0
            else:
                data[sentences[i-1]] = float(sentences[i])
    return data


def main():
    word2vec = read_results("word2vec.txt")
    fasttext = read_results("fasttext.txt")
    glove = read_results("glove.txt")
    plot(word2vec, fasttext, glove)


if __name__ == "__main__":
    main()

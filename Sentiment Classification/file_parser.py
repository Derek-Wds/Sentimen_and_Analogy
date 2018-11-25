import json
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
from util import split_sentence


# function to get all dataset1 files' paths and store them in a dictionary of list
def get_file_path():

    mypath1 = "dataset1/txt_sentoken/neg"
    mypath2 = "dataset1/txt_sentoken/pos"

    neg_files = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
    pos_files = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]

    result = {"neg":[], "pos":[]}

    # add file path for negative data
    for file_name in neg_files:
        file_path = mypath1 + "/" + file_name
        result["neg"].append(file_path)

    # add file path for negative data
    for file_name in pos_files:
        file_path = mypath2 + "/" + file_name
        result["pos"].append(file_path)
    
    return result



# function to read file contents and store them in a dictionary: index -> words list without stop words, save in to json files separately
def read_dataset1_files():
    file_list = get_file_path()
    neg_list = file_list["neg"]
    pos_list = file_list["pos"]
    
    neg_result = dict()
    pos_result = dict()

    idx = 0
    for fp in neg_list:
        words = list()
        with open(fp, "r") as f:
            lines = f.readlines()
            for line in lines:
                words.extend(split_sentence(line))
        neg_result[idx] = words
        idx += 1
    
    with open("temp_data/dataset1_neg_words.json", "w") as f:
        json.dump(neg_result, f)


    idx = 0
    for fp in pos_list:
        words = list()
        with open(fp, "r") as f:
            lines = f.readlines()
            for line in lines:
                words.extend(split_sentence(line))
        pos_result[idx] = words
        idx += 1

    with open("temp_data/dataset1_pos_words.json", "w") as f:
        json.dump(pos_result, f)
    
    return 0

# function to read file from dataset2 and store them in a dictionary as above
def read_dataset2_files():
    neg_result = dict()
    pos_result = dict()

    # negative
    with open("dataset2/rt-polaritydata/rt-polarity.neg", "r", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        idx = 0
        for line in lines:
            words = split_sentence(line)
            neg_result[idx] = words
            idx += 1
    
    with open("temp_data/dataset2_neg_words.json", "w") as f:
        json.dump(neg_result, f)

    # positive
    with open("dataset2/rt-polaritydata/rt-polarity.pos", "r", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        idx = 0
        for line in lines:
            words = split_sentence(line)
            pos_result[idx] = words
            idx += 1

    with open("temp_data/dataset2_pos_words.json", "w") as f:
        json.dump(pos_result, f)
    
    return 0

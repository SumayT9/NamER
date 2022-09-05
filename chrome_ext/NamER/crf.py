from datasets import load_dataset
import re
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import collections
import nltk
import joblib

def calc_shape(word):
    """
    Calculates the shape of a word
    """
    if word is None:
        return None
    t1 = re.sub('[A-Z]', 'X',word)
    t2 = re.sub('[a-z]', 'x', t1)
    return re.sub('[0-9]', 'd', t2)


def extract_features1(sentence, i, parse_chunk, postag):
    """
    Extracts the features necessary to train CRF1

    inputs: sentence: tokenize sentence
            i:index of word to be processed in sentence
            parse_chunk: the shallow parse chunk of a word
            postag: part of speech tag for a word

    output: a feature dict for the word to be processed
    """
    word = sentence[i]
    prev_word = ""
    if i > 0:
        prev_word = sentence[i-1]
    next_word = ""
    if i < len(sentence) - 1:
        next_word = sentence[i+1]
    features = {
        'word': word,
        'prev_word': prev_word,
        'next_word': next_word,
        'shape': calc_shape(word),
        'parse_chunk': parse_chunk,
        'word_before_shape' : calc_shape(prev_word),
        'word_after_shape' : calc_shape(next_word),
        'word_exists_before': int(word in sentence[max(i - 5, 0): i]),
        'word_exists_after' : int(word in sentence[i+1: min(i+5, len(sentence))]),
        'last_bigram' : word[-2:],
        'last_trigram' : word[-3:],
        'postag': postag,
    }
    return features

def create_word_map(data, model, chunk=True):
    """
    Creates a word map to make extraction of document-level features faster for second CRF training

    inputs: data: dataset, this method is currently only applicable to HuggingFace's CONLL2003 data
            model: CRF1 model
            chunk: whether to include shallow parse chunk or not, in real-world data calculating the chunk in a way
                   consistent with the CONLL2003 format would have taken too long so made it optional

    output: word map with document level features
    """
    mapping = {}
    for i in range(len(data)):
        sentence = data[i]["tokens"]
        beginnings = {}
        for j in range(len(sentence)):
            if sentence[j] not in mapping:
                mapping[sentence[j]] = {"labels":[], "sequences":[], "sequence_labels" : {}}
            pos_tag = data[i]["pos_tags"][j]
            parse_chunk = 23
            if chunk:
                parse_chunk = data[i]["chunk_tags"][j]
            label = model.predict(extract_features1(sentence, j, parse_chunk, pos_tag))
            if len(label[0][0]) > 2:
                newlabel = label[0][0][2:]
                if label[0][0].startswith("I") and label[0][0] in beginnings:
                    start = beginnings[newlabel].popleft()
                    mapping[sentence[j]].append((start,j+1, (" ").join(sentence[start:j+1]), newlabel))
                else:
                    beginnings[newlabel] = collections.deque([])
                    if label[0][0].startswith("B"):
                        beginnings[newlabel].append(j)
            mapping[sentence[j]]["labels"].append(newlabel)

    for word in mapping.keys():
        mapping[word]["majority_label"] = max(mapping[word]["labels"], key=lambda x: mapping[word]["labels"].count(x))
        sequence_map = {}
        for sequence in mapping[word]["sequences"]:
            if sequence[2] not in sequence_map:
                sequence_map[sequence[2]] = []
            sequence_map[sequence].append(sequence[3])
        for sequence in sequence_map.keys():
            mapping["word"]["sequence_labels"][sequence] = max(sequence_map[sequence], key = lambda x : sequence_map[sequence].count(x))
    return mapping

def extract_features2(sentence, curr_features, map):
    """
    Extracts document-level features used to train second-level CRF

    inputs: sentence: tokenized sentence
            curr_features: feature dict used for CRF1, document-level features are added to this
            map: word map with document-level information

    output: feature dict for CRF2
    """
    word = curr_features["word"]
    curr_features["majority_label"] = ""
    sequences_including = []
    if word in map:
        curr_features["majority_label"] = map[word]["majority_label"]
        sequences_including = map[word]["sequences"]
    
    joined = (" ").join(sentence)
    super_labels = []
    sequence_to_label = {}
    for sequence in sequences_including:
        super_labels.append(sequence[3])
        if sequence[2] in joined and word in map:
            curr_features["sequence_label"] = map[word]["sequence_labels"][sequence[2]]
        if sequence[2] not in sequence_to_label:
            sequence_to_label[sequence[2]] = []
        sequence_to_label[sequence[2]].append(sequence[3])

    super_label = ""
    if super_labels:
        super_label = max(super_labels, key=lambda x: super_labels.count(x))
    curr_features["superentity_majority_label"] = super_label
    return curr_features

def generate_data(data, chunk=True):
    """
    Generates data to train CRF1

    inputs: data: dataset, this method is currently only applicable to HuggingFace's CONLL2003 data
            chunk: whether to include shallow parse chunk or not, in real-world data calculating the chunk in a way
                   consistent with the CONLL2003 format would have taken too long so made it optional

    output: list of lists of feature dicts for CRF1 and list of lists of corresponding labels
    """
    X, y = [], []
    label_map = {0:'O', 1:'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    for i in range(len(data)):
        sentence = data[i]["tokens"]
        currX, currY = [],[]
        for j in range(len(sentence)):
            pos_tag = data[i]["pos_tags"][j]
            parse_chunk = 23
            if chunk:
                parse_chunk = data[i]["chunk_tags"][j]
            currX.append(extract_features1(sentence, j, parse_chunk, pos_tag))
            currY.append(label_map[data[i]["ner_tags"][j]])
        X.append(currX)
        y.append(currY)
    return X, y


def generate_data2(data, word_map, chunk=True):
    """
    Generates data to train CRF2

    inputs: data: dataset, this method is currently only applicable to HuggingFace's CONLL2003 data
            word_map: word map with document-level features
            chunk: whether to include shallow parse chunk or not, in real-world data calculating the chunk in a way
                   consistent with the CONLL2003 format would have taken too long so made it optional

    output: list of lists of feature dicts for CRF2 and list of lists of corresponding labels
    """
    X, y = [], []
    label_map = {0:'O', 1:'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    for i in range(len(data)):
        sentence = data[i]["tokens"]
        currX, currY = [],[]
        for j in range(len(sentence)):
            pos_tag = data[i]["pos_tags"][j]
            parse_chunk = 23
            if chunk:
                parse_chunk = data[i]["chunk_tags"][j]
            features_1 = extract_features1(sentence, j, parse_chunk, pos_tag)
            currX.append(extract_features2(sentence, features_1, word_map))
            currY.append(label_map[data[i]["ner_tags"][j]])
        X.append(currX)
        y.append(currY)
    return X, y

def inference_data(text_data, model):
    """
    Processes raw string data for model inference

    inputs: text_data: raw string data
            model: crf1 model

    output: list of lists of feature dicts for inference by crf2
    """

    data = nltk.sent_tokenize(text_data)
    data = [nltk.word_tokenize(sent) for sent in data]
    pos_tag_map = {
        '"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
        'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
        'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
        'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
        'WP': 44, 'WP$': 45, 'WRB': 46
    }

    X_out = []
    mapping = {}
    for sent in data:
        for i in range(len(sent)):
            beginnings = {}
            if sent[i] not in mapping:
                mapping[sent[i]] = {"labels":[], "sequences":[], "sequence_labels" : {}}
            pos_tag = pos_tag_map[nltk.pos_tag(sent)[i][1]]
            label = model.predict(extract_features1(sent, i, 23, pos_tag))
            if len(label[0][0]) > 2:
                newlabel = label[0][0][2:]
                if label[0][0].startswith("I") and label[0][0] in beginnings:
                    start = beginnings[newlabel].popleft()
                    mapping[sent[i]].append((start,i+1, (" ").join(sent[start:i+1]), newlabel))
                else:
                    beginnings[newlabel] = collections.deque([])
                    if label[0][0].startswith("B"):
                        beginnings[newlabel].append(i)
                    mapping[sent[i]]["labels"].append(newlabel)

    for word in mapping.keys():
        mapping[word]["majority_label"] = max(mapping[word]["labels"], key=lambda x: mapping[word]["labels"].count(x))
        sequence_map = {}
        for sequence in mapping[word]["sequences"]:
            if sequence[2] not in sequence_map:
                sequence_map[sequence[2]] = []
            sequence_map[sequence].append(sequence[3])
        for sequence in sequence_map.keys():
            mapping["word"]["sequence_labels"][sequence] = max(sequence_map[sequence], key = lambda x : sequence_map[sequence].count(x))

    X_out = []
    for sent in data:
        X_sent = []
        for i in range(len(sent)):
            curr_features = extract_features1(sent, i, 23, pos_tag_map[nltk.pos_tag(sent)[i][1]])
            X_sent.append(extract_features2(sent, curr_features, mapping))
        X_out.append(X_sent)

    return X_out
            
def save_model(model, path):
    """
    Saves model to path provided
    """
    joblib.dump(model, path)



def perform_inference(text, crf1, crf2):
    """
    Performs inference on raw text data
    inputs: text: raw text data
            crf1: CRF1 model
            crf2: CRF2 model
    
    output: dict of results, format- Category: [(word1,sentence number, word number of that sentence), (word2, sentence number...)]

    """
    out = {"People":[], "Locations":[], "Organizations": [], "Other": []}
    data = inference_data(text, crf1)
    pred = crf2.predict(data)
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if "PER" in pred[i][j]:
                out["People"].append((data[i][j]["word"], i, j))
            elif "LOC" in pred[i][j]:
                out["Locations"].append((data[i][j]["word"], i, j))
            elif "ORG" in pred[i][j]:
                out["Organizations"].append((data[i][j]["word"], i, j))
            elif "MISC" in pred[i][j]:
                out["Other"].append((data[i][j]["word"], i, j))
    return out

    


if __name__ == "__main__":
    # dataset = load_dataset("conll2003")
    # print("Generating Train data")
    # X_train, y_train = generate_data(dataset["train"], chunk=False)
    # print("Generating test data")
    # X_test, y_test = generate_data(dataset["test"], chunk=False)
    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     c1=0.1,
    #     c2=0.1,
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )  
    # print("Training models")
    # crf.fit(X_train, y_train)
    # labels = list(crf.classes_)
    # print("Predicting")
    # y_pred = crf.predict(X_test)
    # f1_old = metrics.flat_f1_score(y_test, y_pred,
    #                   average='weighted', labels=labels)

    # print(f1_old)
    # save_model(crf, "crf1_no_chunk")

    # print("Creating word map")
    # word_map = create_word_map(dataset["train"], crf, chunk=False)

    # crf2 = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     c1=0.1,
    #     c2=0.1,
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )  
    # print("Generating Train data")
    # X_train, y_train = generate_data2(dataset["train"], word_map, chunk=False)
    # print("Generating test data")
    # X_test, y_test = generate_data2(dataset["test"], word_map, chunk=False)

    # crf2.fit(X_train, y_train)
    # labels = list(crf2.classes_)
    # print("Predicting")
    # y_pred = crf2.predict(X_test)
    # labels = list(crf.classes_)
    # f1_new = metrics.flat_f1_score(y_test, y_pred,
    #                   average='weighted', labels=labels)
    # print(f1_new)
    # save_model(crf2, "crf2_no_chunk")
    # crf1 = joblib.load("crf1")
    # crf2 = joblib.load("crf2")

    # print("Creating map")
    # word_map = create_word_map(dataset["train"], crf, chunk=False)
    # print("Generating data")
    # X_test, y_test = generate_data2(dataset["test"], word_map, ch)
    # print("done generating data")
    # y_pred = crf2.predict(X_test)
    # labels = list(crf2.classes_)
    # f1_new = metrics.flat_f1_score(y_test, y_pred,
    #                   average='weighted', labels=labels)
    # print(f1_new)


    crf1 = joblib.load("crf1_no_chunk")
    crf2 = joblib.load("crf2_no_chunk")

    test_sent = """(CNN) A man has been charged in connection with the disappearance of a Memphis teacher investigators believe was abducted while she was out for a jog Friday morning, police said.
Cleotha Abston, 38, was charged with especially aggravated kidnapping and tampering with evidence, according to a tweet from the Memphis Police Department posted early Sunday.
"Eliza Fletcher has not been located at this time. MPD Investigators and officers, along with our local and federal partners, continue searching for Mrs. Fletcher," the post said.
Fletcher, 34, was jogging around 4:30 a.m. Friday when an unidentified person approached her, police have said. She was forced into a mid-sized dark SUV and taken from the scene, police said.
"""
    print("generating inference data")
    print(perform_inference(test_sent, crf1, crf2))
    # X_pred = inference_data(test_sent, crf1)
    # pred = crf2.predict(X_pred)
    # data = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(test_sent)]
    # for i in range(len(data)):
    #     for j in range(len(data[i])):
    #         print(data[i][j], pred[i][j])

    




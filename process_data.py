import torch
from torch.utils.data import Dataset

from datasets import load_dataset

dataset = load_dataset("conll2003")
print("done")
label_to_query = {
    1:"Find people in the text", 
    3:"Find organizations in the text, including companies, agencies and institutions.",
    5:"Find locations in the text, including nongeographical locations, mountain ranges and bodies of water",
    7:"Find miscellaneous entities in the text, including events, nationalities, products, or works of art",
}

N_train = len(dataset["train"])
with open("training_data.csv", "w") as training_file:
    training_file.write("Question,Answer,Context")
    for i in range(N_train):
        cur = -5
        current_start = 0
        current_end = 0
        for j in range(len(dataset["train"][i]["tokens"])):
            if dataset["train"][i]["ner_tags"][j] in label_to_query:
                if current_end - current_start > 0:
                    training_file.write(label_to_query[cur] + "," + 
                    (" ").join(dataset["train"][i]["tokens"][current_start:current_end]) + 
                    "," + (" ").join(dataset["train"][i]["tokens"]) + "\n")
                current_start = j
                current_end  = j + 1
                cur = dataset["train"][i]["ner_tags"][j]
            elif dataset["train"][i]["ner_tags"][j] == cur + 1:
                current_end += 1
            else:
                if current_end - current_start > 0:
                    training_file.write(label_to_query[cur] + "," + 
                    (" ").join(dataset["train"][i]["tokens"][current_start:current_end]) + 
                    "," + (" ").join(dataset["train"][i]["tokens"]) + "\n")
                current_start = 0
                current_end = 0
                cur = -5
            









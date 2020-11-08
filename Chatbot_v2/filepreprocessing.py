from hparams import *
import json
import pandas as pd


with open(PATH + before_filename, 'r', encoding = 'utf-8') as json_file:
    json_data = json.loads(json_file.read())

sentence_list = [list(json_data[i]['talk']['content'].values()) for i in range(len(json_data))]

a = 0
b = 1
person_1 = []
person_2 = []
for i in range(len(sentence_list)):
    for ii in range(3):
        person_1.append(sentence_list[i][a])
        person_2.append(sentence_list[i][b])
        a += 2
        b += 2
        if a == 6:
            a = 0
        if b == 7:
            b = 1

speech_data_df = pd.DataFrame({'Input': person_1, 'Output': person_2})

speech_data_df.to_csv(PATH + after_filename, index = False, header = False, sep = '\t')
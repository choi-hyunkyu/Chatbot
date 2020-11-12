from network import *
from hparams import *

import re
import unicodedata
import itertools
import torch

def printLines(file, n=10):
    with open(file, 'r', encoding = 'utf-8') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 소문자로 만들고, 공백을 넣고, 알파벳 외의 글자를 제거합니다
def normalizeString(s):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣 ^☆; ^a-zA-Z.!?;0-9]+')
    result = hangul.sub('', s)
    return result

# 질의/응답 쌍을 읽어서 voc 객체를 반환합니다
def readVocs(datafile, corpus_name, Voc):
    print("Reading lines...")
    # 파일을 읽고, 쪼개어 lines에 저장합니다
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # 각 줄을 쪼개어 pairs에 저장하고 정규화합니다
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# 문장의 쌍 'p'에 포함된 두 문장이 모두 MAX_LENGTH라는 기준보다 짧은지를 반환합니다
def filterPair(p):
    # EOS 토큰을 위해 입력 시퀀스의 마지막 단어를 보존해야 합니다
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# 조건식 filterPair에 따라 pairs를 필터링합니다
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# 앞에서 정의한 함수를 이용하여 만든 voc 객체와 리스트 pairs를 반환합니다
def loadPrepareData(corpus, corpus_name, datafile, save_dir, Voc):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name, Voc)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    # MIN_COUNT 미만으로 사용된 단어는 voc에서 제외합니다
    voc.trim(MIN_COUNT)
    # 제외할 단어가 포함된 경우를 pairs에서도 제외합니다
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # 입력 문장을 F검사합니다
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 출력 문장을 검사합니다
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # 입출력 문장에 제외하기로 한 단어를 포함하지 않는 경우만을 남겨둡니다
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# 입력 시퀀스 텐서에 패딩한 결과와 lengths를 반환합니다
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# 패딩한 목표 시퀀스 텐서, 패딩 마스크, 그리고 최대 목표 길이를 반환합니다
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# 입력 배치를 이루는 쌍에 대한 모든 아이템을 반환합니다
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

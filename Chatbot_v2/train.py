from hparams import *
from usegpu import UseGPU
from preprocessing import *
from network import *
from setting import *

import os
import torch
import torch.optim as optim

'''
gpu
'''
device = UseGPU()


'''
데이터 로딩
'''
corpus_name = corpus_name
corpus = os.path.join("data", corpus_name)
textfilename = after_filename
datafile = os.path.join(corpus, textfilename)


'''
데이터 형태
'''
# printLines(os.path.join(corpus, textfilename))


'''
빈도수 낮은 단어 제거
'''
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir, Voc)
pairs = trimRareWords(voc, pairs, MIN_COUNT)


'''
학습 세팅
'''
# loadFilename이 제공되는 경우에는 모델을 불러옵니다
if loadFilename:
    # 모델을 학습할 때와 같은 기기에서 불러오는 경우
    checkpoint = torch.load(loadFilename)
    # GPU에서 학습한 모델을 CPU로 불러오는 경우
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# 단어 임베딩을 초기화합니다
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# 인코더 및 디코더 모델을 초기화합니다
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# 적절한 디바이스를 사용합니다
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


'''
학습
'''
# Dropout 레이어를 학습 모드로 둡니다
encoder.train()
decoder.train()

# Optimizer를 초기화합니다
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# cuda가 있다면 cuda를 설정합니다
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
    
# 학습 단계를 수행합니다
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)

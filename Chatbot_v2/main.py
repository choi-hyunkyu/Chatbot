from hparams import *
from network import *
from setting import *

import os

corpus = os.path.join("data", corpus_name)
save_dir = os.path.join("data", "save")
datafile = os.path.join(corpus, after_filename)
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir, Voc)
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                            '{}_checkpoint_v1.tar'.format(checkpoint_iter))

# loadFilename이 제공되는 경우에는 모델을 불러옵니다
if loadFilename:
    # 모델을 학습할 때와 같은 기기에서 불러오는 경우
    checkpoint = torch.load(loadFilename)
    # GPU에서 학습한 모델을 CPU로 불러오는 경우
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder = checkpoint['en']
    decoder = checkpoint['de']
    encoder_optimizer = checkpoint['en_opt']
    decoder_optimizer = checkpoint['de_opt']
    embedding = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

# 탐색 모듈을 초기화합니다
searcher = GreedySearchDecoder(encoder, decoder)

# 채팅을 시작합니다 (다음 줄의 주석을 제거하면 시작해볼 수 있습니다)
evaluateInput(encoder, decoder, searcher, voc)
'''
Data Preprocessing part
'''
corpus_name = "normal chatbot data"
PATH = './data/normal chatbot data/'
before_filename = 'speech_data.json'
after_filename = 'speech_data.txt'
MAX_LENGTH = 20  # 고려할 문장의 최대 길이
MIN_COUNT = 3    # 제외할 단어의 기준이 되는 등장 횟수

'''
Network part
'''
# 기본 단어 토큰 값
PAD_token = 0  # 짧은 문장을 채울(패딩, PADding) 때 사용할 제로 토큰
SOS_token = 1  # 문장의 시작(SOS, Start Of Sentence)을 나타내는 토큰
EOS_token = 2  # 문장의 끝(EOS, End Of Sentence)을 나태는 토큰

# 모델을 설정합니다
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1

'''
Train part
'''
batch_size = 64
# 불러올 checkpoint를 설정합니다. 처음부터 시작할 때는 None으로 둡니다.
loadFilename = None
checkpoint_iter = 5000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))

# 학습 및 최적화 설정
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 5000
print_every = 1
save_every = 500

'''
evaluation

pretrained_model_PATH = './data/save/' + model_name + '/' + corpus_name + '/' + str(encoder_n_layers) + '-' + str(decoder_n_layers) + '_' + str(hidden_size) + '/'
model_checkpoint = '5000' + '_' + 'checkpoint_v1' + '.tar'
model_PATH = pretrained_model_PATH + model_checkpoint
'''
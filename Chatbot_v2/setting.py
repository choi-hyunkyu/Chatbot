from usegpu import UseGPU
from hparams import *
from preprocessing import *
from usegpu import UseGPU

import random
import os
import torch
import torch.nn as nn

device = UseGPU()

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # 제로 그라디언트
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # device 옵션을 설정합니다
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # 변수를 초기화합니다
    loss = 0
    print_losses = []
    n_totals = 0

    # 인코더로 포워드 패스를 수행합니다
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # 초기 디코더 입력을 생성합니다(각 문장을 SOS 도큰으로 시작합니다)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # 디코더의 초기 은닉 상태를 인코더의 마지막 은닉 상태로 둡니다
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 이번 반복에서 teacher forcing을 사용할지를 결정합니다
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 배치 시퀀스를 한 번에 하나씩 디코더로 포워드 패스합니다
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing 사용: 다음 입력을 현재의 목표로 둡니다
            decoder_input = target_variable[t].view(1, -1)
            # 손실을 계산하고 누적합니다
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing 미사용: 다음 입력을 디코더의 출력으로 둡니다
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # 손실을 계산하고 누적합니다
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 역전파를 수행합니다
    loss.backward()

    # 그라디언트 클리핑: 그라디언트를 제자리에서 수정합니다
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 모델의 가중치를 수정합니다
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # 각 단계에 대한 배치를 읽어옵니다
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # 초기화
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # 학습 루프
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # 배치에서 각 필드를 읽어옵니다
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # 배치에 대해 학습을 한 단계 진행합니다
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # 경과를 출력합니다
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Checkpoint를 저장합니다
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # 인코더 모델로 입력을 포워드 패스합니다
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 인코더의 마지막 은닉 레이어가 디코더의 첫 번째 은닉 레이어의 입력이 되도록 준비합니다
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # 디코더의 첫 번째 입력을 SOS_token으로 초기화합니다
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # 디코더가 단어를 덧붙여 나갈 텐서를 초기화합니다
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # 반복적으로 각 단계마다 하나의 단어 토큰을 디코딩합니다
        for _ in range(max_length):
            # 디코더로의 포워드 패스를 수행합니다
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 가장 가능성 높은 단어 토큰과 그 softmax 점수를 구합니다
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 토큰과 점수를 기록합니다
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 현재의 토큰을 디코더의 다음 입력으로 준비시킵니다(차원을 증가시켜서)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 단어 토큰과 점수를 모아서 반환합니다
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### 입력 시퀀스를 배치 형태로 만듭니다
    # 단어 -> 인덱스
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # lengths 텐서를 만듭니다
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 배치의 차원을 뒤집어서 모델이 사용하는 형태로 만듭니다
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 적절한 디바이스를 사용합니다
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # searcher를 이용하여 문장을 디코딩합니다
    tokens, scores = searcher(input_batch, lengths, max_length)
    # 인덱스 -> 단어
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 입력 문장을 받아옵니다
            input_sentence = input('> ')
            # 종료 조건인지 검사합니다
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 문장을 정규화합니다
            input_sentence = normalizeString(input_sentence)
            # 문장을 평가합니다
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 응답 문장을 형식에 맞춰 출력합니다
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
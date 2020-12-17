# Based on: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import random
import torch
import pdb
from torch.autograd import Variable
import numpy as np
from components.data.common import cuda_if_gpu
from components.constants import BOS_ID, EOS_ID, PAD_ID
from components.model import E2ESeq2SeqModel
from components.model.modules.encoders.enc_transformer import make_encoder
from components.model.modules.decoders.dec_transformer import make_decoder


class E2ETransformer(E2ESeq2SeqModel):
    def set_encoder(self):
        self.hit_input = self.config["hit_input"]
        self.model_size = self.config["encoder_params"]["hidden_size"]
        encoder_params = self.config["encoder_params"]
        self.encoder = make_encoder(encoder_params)
        self.nos_position=self.config["nos_position"]
        self.nos_predict_strategy = self.config["nos_predict_strategy"]
        self.nos_predict_sent_num = self.config["nos_predict_sent_num"]
        self.nos_option = self.config["nos_option"]        

    def set_decoder(self):
        decoder_rnn_params = self.config["decoder_params"]
        self.decoder = make_decoder(decoder_rnn_params, self.tgt_vocab_size)

    def forward(self, datum):
        """
        Run the model on one data instance.
        :param datum:
        :return:
        """

        batch_x_var, batch_y_var = datum        
        encoder_input_embedded = self.embedding(batch_x_var)
        encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded).transpose(0,1)
        
        #CNE
        if self.nos_option == 1:
            batch_y_var = batch_y_var.transpose(0, 1)
            batch_size = batch_y_var.size(0)
            bos_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
            decoder_input = torch.cat([bos_input.unsqueeze(1), batch_y_var[:,:-1]], dim=-1)
            decoder_input_embedded = self.embedding_mat(decoder_input)
            decoder_input_nos=cuda_if_gpu(Variable(torch.LongTensor(np.zeros((decoder_input.size(0),1)))))
            
            for i in range(decoder_input.size(0)):
                for j in range(decoder_input.size(1)):
                    if decoder_input[i,j]==41:  #if there is a ".". sentence count+1
                        decoder_input_nos[i]+=1

            if self.nos_position == "encoder":
                encoder_input_embedded_nos = self.embedding_mat_nos(decoder_input_nos)
                encoder_input_embedded = encoder_input_embedded+encoder_input_embedded_nos
                                                
            if self.nos_position == "decoder":
                #decoder nos embedding
                decoder_input_embedded_nos = self.embedding_mat_nos(decoder_input_nos)
                decoder_input_embedded = decoder_input_embedded + decoder_input_embedded_nos        
        # PAG
        elif self.nos_option == 2:
            nos_list = [] 
            batch_y_var = batch_y_var.transpose(0, 1)
            batch_size = batch_y_var.size(0)
            bos_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))

            nos_num = (batch_y_var == 41).sum(dim=-1)
            nos_input = nos_num + self.tgt_vocab_size - 7
            
            decoder_input = torch.cat([bos_input.unsqueeze(1), nos_num.unsqueeze(1), batch_y_var[:,:-1]], dim=-1)
            decoder_input_embedded = self.embedding_mat(decoder_input)
        # No nos
        else:
            batch_y_var = batch_y_var.transpose(0, 1)
            batch_size = batch_y_var.size(0)
            bos_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
            decoder_input = torch.cat([bos_input.unsqueeze(1), batch_y_var[:,:-1]], dim=-1)
            decoder_input_embedded = self.embedding_mat(decoder_input)

        encoder_outputs = self.encoder(encoder_input_embedded, None)
        tgt_mask = (decoder_input != PAD_ID).unsqueeze(1)
        tgt_mask = tgt_mask & self.subsequent_mask(decoder_input.size(-1)).type_as(tgt_mask)
        logits = self.decoder(decoder_input_embedded, encoder_outputs, None, tgt_mask)
        return logits.transpose(0,1)

    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)

    def embedding(self, batch_x_var):
        # Embedding lookup
        if self.hit_input:
            seq_len, batch_size = batch_x_var.size()
            seq_len = seq_len // 2
            batch_x_var = batch_x_var.view(-1, 2, batch_size).transpose(-1, -2)

        encoder_input_embedded = self.embedding_lookup(batch_x_var)  # SL x B x E

        if self.hit_input:
            encoder_input_embedded = encoder_input_embedded.view(seq_len, batch_size, -1)
        
        return encoder_input_embedded

    def predict(self, input_var, K): 

        # Embedding lookup
        encoder_input_embedded = self.embedding(input_var).transpose(0, 1)
        
        if self.nos_option == 1:
            if self.nos_predict_strategy == "fix":
                nos = cuda_if_gpu(torch.tensor(self.nos_predict_sent_num))
                nos_embedding = self.embedding_mat_nos(nos)
            elif self.nos_predict_strategy == "uniform_random":
                nos = cuda_if_gpu(torch.randint(1,6,(1,)))
                nos_embedding = self.embedding_mat_nos(nos)
            elif self.nos_predict_strategy == "distributive_random":
                prob_list = [5.82467e-1, 3.44111e-1, 6.10570e-2, 1.072303e-2, 1.4741196e-3, 1.664328683-4]
                bag=np.ones((1000,1))
                bias=0
                for i in range(len(prob_list)):
                    cnt=int(round(1000*prob_list[i]))
                    for j in range(cnt):
                        bag[j+bias]=i+1
                    bias=bias+cnt    
                k=random.choice(bag)
                k=int(k[0])
                nos = cuda_if_gpu(Variable(torch.tensor(k)))
                nos_embedding = self.embedding_mat_nos(nos)
            elif self.nos_predict_strategy == "threshold":
                length=0
                for var_id in input_var:
                    if var_id!=3 and var_id!=2:
                        length+=1
                k = 1 if length<=5 else 2
                nos = cuda_if_gpu(Variable(torch.tensor(k)))
                nos_embedding = self.embedding_mat_nos(nos)
            elif self.nos_predict_strategy == "weight_embedding":
                average = [1.0865019, 1.2313611, 1.4140642, 1.6457017, 1.8368207, 1.92149088]
                length=0
                for var_id in input_var:
                    if var_id!=3 and var_id!=2:
                        length+=1
                ids=length-3
                score=average[ids]
                left_weight=score-1
                right_weight=2-score
           
                k1=cuda_if_gpu(Variable(torch.tensor(1)))
                k2=cuda_if_gpu(Variable(torch.tensor(2)))
            
                nos_embedding_1=self.embedding_mat_nos(k1)
                nos_embedding_2=self.embedding_mat_nos(k2)
                nos_embedding=(right_weight*nos_embedding_1+left_weight*nos_embedding_2)/2        
            
            if self.nos_position == "encoder":
                encoder_input_embedded = encoder_input_embedded + nos_embedding


        # Encode
        encoder_outputs = self.encoder(encoder_input_embedded, None)

        # Decode
        iy = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID]))).unsqueeze(0)
        topk_dec_seqs = [[iy, 0.0, [BOS_ID]] ]
        curr_dec_idx = 0 

        while curr_dec_idx <= self.max_tgt_len:
            all_seqs = []
            iys, enc_outs = [], []
            for i in range(len(topk_dec_seqs)):
                if topk_dec_seqs[i][-1][-1] == EOS_ID:
                    all_seqs.append(topk_dec_seqs[i])
                    continue
                enc_outs.append(encoder_outputs)
                iys.append(topk_dec_seqs[i][0])

            if len(iys) == 0:
                break
            dec_input_var = torch.cat(iys, dim=0)
            enc_outs = torch.cat(enc_outs, dim=0)
            tgt_mask = (dec_input_var != PAD_ID).unsqueeze(1)
            tgt_mask = tgt_mask & self.subsequent_mask(dec_input_var.size(-1)).type_as(tgt_mask)

            #import pdb
            #pdb.set_trace()
    
            prev_y = self.embedding_mat(dec_input_var)
            if self.nos_option == 1 and self.nos_position== "decoder":
                prev_y = prev_y + nos_embedding

            decoder_output = self.decoder(prev_y, enc_outs, None, tgt_mask)
                #attn_w.append(decoder_attention.data)
    
            topval, topidx = decoder_output[:,-1,:].data.topk(K)

            pos = -1
            for i in range(len(topk_dec_seqs)):
                if topk_dec_seqs[i][-1][-1] == EOS_ID:
                    continue
                pos += 1

                for j in range(K):
                    idx = topidx[pos][j].item()
                    score = topval[pos][j].item()

                    new_seq = topk_dec_seqs[i][2] + [idx]
                    new_score = topk_dec_seqs[i][1] + score
                    new_iy = torch.cat([topk_dec_seqs[i][0], topidx[pos][j].view(-1,1)], dim=-1)
                    all_seqs.append([new_iy, new_score, new_seq])

            sort_f = lambda x:x[1]
            topk_dec_seqs = sorted(all_seqs, key=sort_f, reverse=True)[:K]
            curr_dec_idx += 1
    
        return topk_dec_seqs[0][0].squeeze(0)[1:], None

    def predict_dev(self, input_var, nos):

        # Embedding lookup
        encoder_input_embedded = self.embedding(input_var).transpose(0, 1)
        nos=cuda_if_gpu(Variable(torch.tensor(nos)))
        nos_embedding=self.embedding_mat_nos(nos)
        if self.nos_position=="encoder":
            encoder_input_embedded = encoder_input_embedded + nos_embedding

        # Encode
        encoder_outputs = self.encoder(encoder_input_embedded, None)

        # Decode
        dec_ids, attn_w = [], [1]
        curr_token_id = BOS_ID
        curr_dec_idx = 0
        dec_input_ids = [curr_token_id]
        
        while (curr_token_id != EOS_ID and curr_dec_idx <= self.max_tgt_len):
            dec_input_var = cuda_if_gpu(Variable(torch.LongTensor(dec_input_ids))).unsqueeze(0)
            tgt_mask = (dec_input_var != PAD_ID).unsqueeze(1)
            tgt_mask = tgt_mask & self.subsequent_mask(dec_input_var.size(-1)).type_as(tgt_mask)

            prev_y = self.embedding_mat(dec_input_var)
            if self.nos_position == "decoder":
                #decoder_embedding
                nos_embedding = self.embedding_mat_nos(nos)
                prev_y=prev_y + nos_embedding
                #decoder_embedding end
            
            decoder_output = self.decoder(prev_y, encoder_outputs, None, tgt_mask)
            #attn_w.append(decoder_attention.data)

            topval, topidx = decoder_output[:,-1,:].data.topk(1)
            curr_token_id = topidx[0][0]
            dec_ids.append(curr_token_id)
            dec_input_ids.append(curr_token_id)

            curr_dec_idx += 1
        #pdb.set_trace()
        return dec_ids, attn_w


component = E2ETransformer

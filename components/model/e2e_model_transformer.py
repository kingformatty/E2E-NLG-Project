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
        self.nos_predict_fix = self.config["nos_predict_fix"]
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
        if self.nos_option == 1:
            #pdb.set_trace()
            batch_y_var = batch_y_var.transpose(0, 1)
            batch_size = batch_y_var.size(0)
            bos_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
            decoder_input = torch.cat([bos_input.unsqueeze(1), batch_y_var[:,:-1]], dim=-1)
            decoder_input_embedded = self.embedding_mat(decoder_input)
            #pdb.set_trace()
            decoder_input_nos=cuda_if_gpu(Variable(torch.LongTensor(np.zeros((decoder_input.size(0),1)))))
            for i in range(decoder_input.size(0)):
                for j in range(decoder_input.size(1)):
                    if decoder_input[i,j]==41:  #if there is a ".". sentence count+1
                        decoder_input_nos[i]+=1
            #pdb.set_trace()
            
            encoder_input_embedded = self.embedding(batch_x_var)
            encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded).transpose(0,1)
            if self.nos_position == "encoder":
                #encoder nos embedding
                encoder_input_embedded_nos = self.embedding_mat_nos(decoder_input_nos)
                encoder_input_embedded=encoder_input_embedded+encoder_input_embedded_nos
                #encoder nos embedding end
        
            encoder_outputs = self.encoder(encoder_input_embedded, None)
        # PAG
        elif self.nos_option == 2:
            nos_list = [] 
            encoder_input_embedded = self.embedding(batch_x_var)
            encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded).transpose(0,1)
            encoder_outputs = self.encoder(encoder_input_embedded, None)

            batch_y_var = batch_y_var.transpose(0, 1)
            batch_size = batch_y_var.size(0)
            bos_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))

            #pdb.set_trace()
            nos_num = (batch_y_var == 41).sum(dim=-1)
            for x in nos_num:
               if x == 1 or x == 0:
                  nos_list.append(3251)
               if x == 2:
                  nos_list.append(3252)
               if x == 3:
                  nos_list.append(3253)
               if x == 4:
                  nos_list.append(3254)
               if x == 5:
                  nos_list.append(3255)
               if x == 6:
                  nos_list.append(3256)

            nos_input = cuda_if_gpu(Variable(torch.LongTensor(nos_list)))
            #pdb.set_trace()
            decoder_input = torch.cat([bos_input.unsqueeze(1), nos_input.unsqueeze(1), batch_y_var[:,:-1]], dim=-1)
            encoder_input_embedded = self.embedding_mat(decoder_input)
        
        if self.nos_position == "decoder":
            #decoder nos embedding
            decoder_input_embedded_nos = self.embedding_mat_nos(decoder_input_nos)
            decoder_input_embedded=decoder_input_embedded+decoder_input_embedded_nos
        #pdb.set_trace()
        tgt_mask = (decoder_input != PAD_ID).unsqueeze(1)
        tgt_mask = tgt_mask & self.subsequent_mask(decoder_input.size(-1)).type_as(tgt_mask)
        pdb.set_trace()
        logits = self.decoder(decoder_input_embedded, encoder_outputs, None, tgt_mask)
        #pdb.set_trace()
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

    def predict(self, input_var):

        # Embedding lookup
        encoder_input_embedded = self.embedding(input_var).transpose(0, 1)
        if self.nos_predict_strategy == "fix":
            nos=cuda_if_gpu(torch.tensor(self.nos_predict_fix))
            nos_embedding=self.embedding_mat_nos(nos)
        elif self.nos_predict_strategy == "uniform_random":
            nos=cuda_if_gpu(torch.randint(1,6,(1,)))
            nos_embedding=self.embedding_mat_nos(nos)
        elif self.nos_predict_strategy == "distributive_random":
            prob_list=[5.82467e-1,3.44111e-1,6.10570e-2,1.072303e-2,1.4741196e-3,1.664328683-4]
            bag=np.ones((1000,1))
            bias=0
            for i in range(len(prob_list)):
                cnt=int(round(1000*prob_list[i]))
                for j in range(cnt):
                    bag[j+bias]=i+1
                bias=bias+cnt    
            k=random.choice(bag)
            k=int(k[0])
            nos=cuda_if_gpu(Variable(torch.tensor(k)))
            nos_embedding=self.embedding_mat_nos(nos)
        elif self.nos_predict_strategy == "threshold":
            length=0
            for var_id in input_var:
                if var_id!=3 and var_id!=2:
                    length+=1
            if length<=5:
                k=1
            else:
                k=2
            nos=cuda_if_gpu(Variable(torch.tensor(k)))
            nos_embedding=self.embedding_mat_nos(nos)
        elif self.nos_predict_strategy == "weight_embedding":
            average = [1.0865019,1.2313611,1.4140642,1.6457017,1.8368207,1.92149088]
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
        #threshold
        #length=0
        #for var_id in input_var:
        #    if var_id!=3 and var_id!=2:
        #        length+=1
        #if length<=5:
        #    k=1
        #else:
        #    k=2
        #threshold end
        #k=cuda_if_gpu(torch.tensor(n))
        #uniform random
        #k=cuda_if_gpu(torch.randint(1,6,(1,)))
        #uniform random end
        #random
        #prob_list=[5.82467e-1,3.44111e-1,6.10570e-2,1.072303e-2,1.4741196e-3,1.664328683-4]
        #bag=np.ones((1000,1))
        #bias=0
        #for i in range(len(prob_list)):
        #    cnt=int(round(1000*prob_list[i]))
        #    for j in range(cnt):
        #        bag[j+bias]=i+1
        #    bias=bias+cnt    
        #k=random.choice(bag)
        #k=int(k[0])
        #k=cuda_if_gpu(Variable(torch.tensor(k)))
        #random end
        
        #weight embedding
        #average = [1.0865019,1.2313611,1.4140642,1.6457017,1.8368207,1.92149088]
        #length=0
        #for var_id in input_var:
        #    if var_id!=3 and var_id!=2:
        #        length+=1
        #ids=length-3
        #score=average[ids]
        #left_weight=score-1
        #right_weight=2-score
       # 
        #k1=cuda_if_gpu(Variable(torch.tensor(1)))
        #k2=cuda_if_gpu(Variable(torch.tensor(2)))
        
        #nos_embedding_1=self.embedding_mat_nos(k1)
        #nos_embedding_2=self.embedding_mat_nos(k2)
        #nos_embedding=(right_weight*nos_embedding_1+left_weight*nos_embedding_2)/2
        #weight embedding end
        
        #nos_embedding=self.embedding_mat_nos(k)
        if self.nos_position == "encoder":
            encoder_input_embedded=encoder_input_embedded+nos_embedding
        # Encode
        encoder_outputs = self.encoder(encoder_input_embedded, None)

        # Decode
        dec_ids, attn_w = [], [1]
        curr_token_id = BOS_ID
        curr_dec_idx = 0
        dec_input_ids = [curr_token_id]
        #k=cuda_if_gpu(Variable(torch.randint(0,3,(1,1))))
        #pdb.set_trace()
        #length=0
        #for var_id in input_var:
        #    if var_id!=3 and var_id!=2:
        #        length+=1
        #if length<=5:
        #    k=1
        #else:
        #    k=2
        #k=cuda_if_gpu(Variable(torch.tensor(k)))
        while (curr_token_id != EOS_ID and curr_dec_idx <= self.max_tgt_len):
            dec_input_var = cuda_if_gpu(Variable(torch.LongTensor(dec_input_ids))).unsqueeze(0)
            tgt_mask = (dec_input_var != PAD_ID).unsqueeze(1)
            tgt_mask = tgt_mask & self.subsequent_mask(dec_input_var.size(-1)).type_as(tgt_mask)

            prev_y = self.embedding_mat(dec_input_var)
            #pdb.set_trace()
            if self.nos_position== "decoder":
            
                prev_y=prev_y+nos_embedding
            decoder_output = self.decoder(prev_y, encoder_outputs, None, tgt_mask)
            #attn_w.append(decoder_attention.data)

            topval, topidx = decoder_output[:,-1,:].data.topk(1)
            curr_token_id = topidx[0][0]
            dec_ids.append(curr_token_id)
            dec_input_ids.append(curr_token_id)

            curr_dec_idx += 1

        return dec_ids, attn_w

    def predict_dev(self, input_var, nos):

        # Embedding lookup
        encoder_input_embedded = self.embedding(input_var).transpose(0, 1)
        nos=cuda_if_gpu(Variable(torch.tensor(nos)))
        nos_embedding=self.embedding_mat_nos(nos)
        if self.nos_position=="encoder":
            encoder_input_embedded = encoder_input_embedded+nos_embedding

        # Encode
        encoder_outputs = self.encoder(encoder_input_embedded, None)

        # Decode
        dec_ids, attn_w = [], [1]
        curr_token_id = BOS_ID
        curr_dec_idx = 0
        dec_input_ids = [curr_token_id]
        #k=cuda_if_gpu(Variable(torch.randint(0,3,(1,1))))
        #nos=cuda_if_gpu(Variable(torch.tensor(nos)))
        
        while (curr_token_id != EOS_ID and curr_dec_idx <= self.max_tgt_len):
            dec_input_var = cuda_if_gpu(Variable(torch.LongTensor(dec_input_ids))).unsqueeze(0)
            tgt_mask = (dec_input_var != PAD_ID).unsqueeze(1)
            tgt_mask = tgt_mask & self.subsequent_mask(dec_input_var.size(-1)).type_as(tgt_mask)

            prev_y = self.embedding_mat(dec_input_var)
            #pdb.set_trace()
            if nos_position == "decoder":
                #decoder_embedding
                nos_embedding = self.embedding_mat_nos(nos)
                prev_y=prev_y+nos_embedding
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

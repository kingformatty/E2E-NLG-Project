# Based on: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import random
import torch
from torch.autograd import Variable

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
        encoder_outputs = self.encoder(encoder_input_embedded, None)

        batch_y_var = batch_y_var.transpose(0, 1)
        batch_size = batch_y_var.size(0)
        bos_input = cuda_if_gpu(Variable(torch.LongTensor([BOS_ID] * batch_size)))
        decoder_input = torch.cat([bos_input.unsqueeze(1), batch_y_var[:,:-1]], dim=-1)
        decoder_input_embedded = self.embedding_mat(decoder_input)
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

    def predict(self, input_var):

        # Embedding lookup
        encoder_input_embedded = self.embedding(input_var).transpose(0, 1)

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
            
            decoder_output = self.decoder(prev_y, encoder_outputs, None, tgt_mask)
            #attn_w.append(decoder_attention.data)

            topval, topidx = decoder_output[:,-1,:].data.topk(1)
            curr_token_id = topidx[0][0]
            dec_ids.append(curr_token_id)
            dec_input_ids.append(curr_token_id)

            curr_dec_idx += 1

        return dec_ids, attn_w


component = E2ETransformer

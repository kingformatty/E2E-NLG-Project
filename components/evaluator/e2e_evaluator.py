import logging
import pdb
from components.constants import NAME_TOKEN, NEAR_TOKEN
from components.data.common import ids2var

logger = logging.getLogger('experiment')


class BaseEvaluator(object):
    """
    Base class containing methods for evaluation of E2E NLG models.
    """

    def __init__(self, config):
        self.config = config or dict()

    def label2snt(self, id2word, ids):
        tokens = [id2word[t] for t in ids]
        return tokens, ' '.join(tokens)

    def predict_one(self, model, src_snt_ids, top_k):
        addEOS = not model.hit_input
        input_var = ids2var(src_snt_ids, -1, 1, addEOS=addEOS)  # batch_size = 1; cudified
        output_ids, attention_weights = model.predict(input_var, top_k)
        return output_ids, attention_weights

    def predict_one_dev(self, model, src_snt_ids, nos):
        addEOS = not model.hit_input
        input_var = ids2var(src_snt_ids, -1, 1, addEOS=addEOS)
        output_ids, attention_weights = model.predict_dev(input_var, nos)
        return output_ids, attention_weights
        
    def evaluate_model(self, model, dev_data, dev_data_tgt):
        """
        Evaluating model on multi-ref data
        :param model:
        :param dev_data:
        :return:
        """

        decoded_ids = []
        decoded_attn_weights = []
        nos_option = model.nos_option
        # Make a prediction on the first input
        if nos_option == 1:
            curr_x_ids = dev_data[0]
            curr_tgt_ids = dev_data_tgt[0]
            nos=0        
            for i in range(len(curr_tgt_ids)):
                if curr_tgt_ids[i]==41:
                    nos+=1

            out_ids, attn_weights = self.predict_one_dev(model, curr_x_ids, nos)
            decoded_ids.append(out_ids)
            decoded_attn_weights.append(attn_weights)

            # Make predictions on the remaining unique (!) inputs
            cnt=1
            for snt_ids in dev_data[1:]:
                curr_tgt_ids = dev_data_tgt[cnt]
                nos=0
                for j in range(len(curr_tgt_ids)):
                    if curr_tgt_ids[j]==41:
                        nos+=1

                if snt_ids == curr_x_ids:
                    continue
                else:
                    out_ids, attn_weights = self.predict_one_dev(model, snt_ids, nos)
                    decoded_ids.append(out_ids)
                    decoded_attn_weights.append(attn_weights)
                    curr_x_ids = snt_ids
                cnt += 1

        else:
            # Make a prediction on the first input
            curr_x_ids = dev_data[0]
            out_ids, attn_weights = self.predict_one(model, curr_x_ids, 1)
            if nos_option == 2:
                decoded_ids.append(out_ids[1:])
            else:
                decoded_ids.append(out_ids)
            decoded_attn_weights.append(attn_weights)
            # Make predictions on the remaining unique (!) inputs
            for snt_ids in dev_data[1:]:

                if snt_ids == curr_x_ids:
                    continue
                else:
                    out_ids, attn_weights = self.predict_one(model, snt_ids, 1)
                    if nos_option == 2:
                        decoded_ids.append(out_ids[1:])
                    else:
                        decoded_ids.append(out_ids)
                    decoded_attn_weights.append(attn_weights)
                    curr_x_ids = snt_ids
        
        return decoded_ids, decoded_attn_weights

    def evaluate_model_test(self, model, test_data):
        """
        Evaluating model on orginal data
        :param model:
        :param dev_data:
        :return:
        """
        nos_option = self.config["model_params"]["nos_option"]
        top_k = self.config["model_params"]["top_k"]
        decoded_ids = []
        decoded_attn_weights = []
        for snt_ids in test_data:
            out_ids, attn_weights = self.predict_one(model, snt_ids, top_k)
            if nos_option == 2:
                decoded_ids.append(out_ids[1:])
            else:
                decoded_ids.append(out_ids)
            decoded_attn_weights.append(attn_weights)

        return decoded_ids, decoded_attn_weights


    def lexicalize_predictions(self, all_tokids, data_lexicalizations, id2word):
        """
        Given model predictions from a model, convert numerical ids to tokens,
        substituting placeholder items (NEAR and NAME) with the values in "data_lexicalizations",
        which we created during the data preprocessing step.

        :param all_tokids:
        :param data_lexicalizations:
        :param id2word:
        :return:
        """

        all_tokens = []

        for idx, snt_ids in enumerate(all_tokids):

            this_snt_toks = []
            this_snt_lex = data_lexicalizations[idx]

            for t in snt_ids[:-1]:  # excluding </s>

                tok = id2word[t.item()]

                if tok == NAME_TOKEN:
                    l = this_snt_lex[0]
                    if not l is None:
                        this_snt_toks.append(l)

                elif tok == NEAR_TOKEN:
                    l = this_snt_lex[1]
                    if not l is None:
                        this_snt_toks.append(l)

                else:
                    this_snt_toks.append(tok)

            all_tokens.append(this_snt_toks)

        return all_tokens


component = BaseEvaluator

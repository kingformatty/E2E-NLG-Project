import os
import logging
import numpy as np
import torch
import torch.nn as nn
from components.utils.visualize import plot_train_progress, plot_lcurve
from components.data.common import cuda_if_gpu
from torch.autograd import Variable

from components.trainer import BaseTrainer

logger = logging.getLogger('experiment')


class E2EMLPTrainer(BaseTrainer):
    def set_train_criterion(self, vocab_size, pad_id):
        """
        NMT Criterion from: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Loss.py
        :return:

        """

        weight = torch.ones(vocab_size)
        weight[pad_id] = 0
        # self.criterion = nn.NLLLoss(weight, size_average=True)
        self.criterion = nn.NLLLoss(weight, reduction="mean")

        if self.use_cuda:
            self.criterion = self.criterion.cuda()

    def train_step(self, model, datum):
        datum = [cuda_if_gpu(Variable(torch.LongTensor(t)).transpose(0, 1))
                 for t in datum]  # [SL x B, TL x B]

        logits = model.forward(datum)  # TL x B x TV
        loss_var = self.calc_loss(logits, datum)  # have to compute log_logits, since using NLL loss
        return loss_var

    def calc_loss(self, logits, datum):
        batch_y_var = datum[1]
        vocab_size = logits.size()[-1]
        logits = logits.contiguous().view(-1, vocab_size)
        if nos_option == 2:
            nos_list = []
            nos_num = (batch_y_var.transpose(0, 1) == 41).sum(dim = -1)
        
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
            nos = cuda_if_gpu(Variable(torch.LongTensor(nos_list)))
            y_nos = torch.cat([nos.unsqueeze(1), batch_y_var.transpose(0,1)], dim=-1).transpose(0,1)
            targets = y_nos.contiguous().view(-1,1).squeeze(1)
 
            loss = self.criterion(logits, targets).contiguous().view(-1, 1).squeeze(1)
        else:
            targets = batch_y_var.contiguous().view(-1,1).squeeze(1)
            loss = self.criterion(logits,targets)  
           
        return loss

    def plot_lcurve(self):
        fig_fname = os.path.join(self.model_dir, "lcurve.pdf")
        title = self.config['modeltype']
        plot_lcurve(self.train_losses, self.dev_losses, img_title=title, save_path=fig_fname, show=False)

    def plot_training_results(self):
        losses = np.asarray([self.train_losses, self.dev_losses]).transpose()
        plot_train_progress(scores=(losses,
                                    self.bleu_scores,
                                    self.nist_scores,
                                    self.cider_scores,
                                    self.rouge_scores,
                                    self.meteor_scores),
                            names=self.get_plot_names(),
                            img_title=self.config['modeltype'],
                            save_path=os.path.join(self.model_dir, "lcurve_scores.pdf"),
                            show=False)

    def get_plot_names(self):
        return [['TrainLoss', 'DevLoss'], 'BLEU', 'NIST', 'CIDEr', 'ROUGE_L', 'METEOR']


component = E2EMLPTrainer

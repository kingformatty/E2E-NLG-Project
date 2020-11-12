from components.model.modules.encoders import EncoderRNN
from components.model import get_GRU_unit


class EncoderGRU(EncoderRNN):
    def __init__(self, config):
        super(EncoderGRU, self).__init__()

        self.config = config
        self.hidden_size = self.config['hidden_size']
        self.rnn = get_GRU_unit(config)


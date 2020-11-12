PAD_TOKEN = '<blank>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
NAME_TOKEN = '<name>'
NEAR_TOKEN = '<near>'

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

START_VOCAB = [PAD_TOKEN,
               BOS_TOKEN,
               EOS_TOKEN,
               UNK_TOKEN,
               ]

MR_FIELDS = ["name", "familyFriendly", "eatType", "food", "priceRange", "near", "area", "customer rating"]
MR_KEYMAP = dict((key, idx) for idx, key in enumerate(MR_FIELDS))


MR_HAV_TOKENS = {"name": ['<s_name>', '<e_name>'],
                 "familyFriendly": ['<s_family>', '<e_family>'], 
                 "eatType": ['<s_eatType>', '<e_eatType>'], 
                 "food": ['<s_food>', '<e_food>'], 
                 "priceRange": ['<s_priceRange>', '<e_priceRange>'], 
                 "near": ['<s_near>', '<e_near>'], 
                 "area": ['<s_area>', '<e_area>'], 
                 "customer rating": ['<s_customerRating>', '<e_customerRating>']
                }



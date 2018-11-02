import logging
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin, WordsRef


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def create_model(args, maxlen, vocab, wordsref):

    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        wn_ = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(WeightedSum), axis=-1, keepdims=True)), K.floatx())
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n))-K.eye(w_n.shape[0])))
        return args.ortho_reg*reg

    vocab_size = len(vocab)
    ##################     Inputs    #################
    sentence_input = Input(shape=(maxlen,), dtype='int32', name='sentence_input')
    neg_input = Input(shape=(args.neg_size, maxlen), dtype='int32', name='neg_input')
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')
    
    ######### Compute sentence representation ########
    e_w = word_emb(sentence_input)

    e_wordsref = word_emb(wordsref)

    y_s = Average()(e_w)

    att_words_ref = WordsRef(name='att_words_ref')([e_wordsref, e_w, y_s])
    z_ss = WeightedSum()([e_w, att_words_ref])
    
    ######## Compute representations of negative instances #######
    e_neg = word_emb(neg_input)
    z_n = Average()(e_neg)




        

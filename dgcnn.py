from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

if __name__ == '__main__':
    config = K.tf.ConfigProto()
    # 关于GPU的设置，我的设备不支持GPU所以就注释掉
    # config.gpu_options.per_process_gup_memory_fraction = 0.7
    session = K.tf.Session(config=config)
    K.set_session(session)


def seq_gather(x):
    """seq是[None,seq_len,s_size]的格式，idxs是[None,1]的格式
    在seq的第i个序列中选出第idxs[i]个向量，最后输出[None,s_size]的向量
    """
    seq,idxs = x
    idxs = K.cast(idxs,'int32')
    batch_idxs = K.arange(0,K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs,1)
    idxs = K.concatenate([batch_idxs,idxs],1)
    return K.tf.gather_nd(seq,idxs)

def seq_maxpool(x):
    """seq是[None,seq_len,s_size]的格式，
    mask是[None,seq_len,1]的格式，先除去mask部分
    然后再做maxpooling
    """
    seq,mask = x
    seq -= (1 - mask)*1e10
    return K.max(seq,1,keepdims=True)

def dilated_gated_conv1d(seq,mask,dilation_rate=1):
    """膨胀门卷积（残差式）
    :param seq:
    :param mask:
    :param dilation_rate:
    :return:
    """
    dim = K.int_shape(seq)[-1]
    h = Conv1D(dim*2,3,padding='same',dilation_rate=dilation_rate)(seq)
    def _gate(x):
        dropout_rate = 0.1
        s,h = x
        g,h = h[:,:,:dim],h[:,:,dim:]
        g = K.in_train_phase(K.dropout(g,dropout_rate),g)
        g = K.sigmoid(g)
        return g*s+(1-g)*h
    seq = Lambda(_gate)([seq,h])
    seq = Lambda(lambda x:x[0] * x[1])([seq,mask])
    return seq

class Attention(Layer):
    """
    多头注意力机制，使用Google的self-attention
    """


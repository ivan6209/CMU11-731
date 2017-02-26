import dynet as dy
#print(dy.__version__)

import numpy as np


class AttentionTranslator():
    def __init__(self, srcDict, tgtDict, embedding_size=64, hidden_size=64, attention_size=64, layer_depth=2, builder=dy.SimpleRNNBuilder, attention='dot',  dropout=None):
        self.useEncoding = True
        self._attention = attention
        
        self.src_w2id, self.src_id2w = srcDict
        self.tgt_w2id, self.tgt_id2w = tgtDict
        self.tgt_vocab_size = len(self.tgt_id2w)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_depth = layer_depth
        self.attention_size = attention_size
        
        # define dynet model
        self.model = dy.Model()
        
        self.l2r_builder = builder(layer_depth, embedding_size, hidden_size, self.model)
        self.r2l_builder = builder(layer_depth, embedding_size, hidden_size, self.model)
        self.dec_builder = builder(layer_depth, embedding_size+2*hidden_size, hidden_size, self.model)
        if dropout is not None:
            self.l2r_builder.set_dropout(dropout)
            self.r2l_builder.set_dropout(dropout)
            self.dec_builder.set_dropout(dropout)
        self.src_lookup = self.model.add_lookup_parameters((len(self.src_id2w), embedding_size))
        self.tgt_lookup = self.model.add_lookup_parameters((len(self.tgt_id2w), embedding_size))
        self.W_ = self.model.add_parameters((self.tgt_vocab_size, hidden_size))
        self.b_ = self.model.add_parameters((self.tgt_vocab_size))
        print('vocabulary size: source=',len(self.src_id2w), '. target=', len(self.tgt_id2w))
        print('hidden_size',hidden_size)
        print('embedding_size', embedding_size)
        print('attention_size', attention_size)
        

        if self._attention.lower() == 'mlp':
        # Note that in the original paper (arxiv 149.0473), attention_size = hidden_size = n
            self.Wa1_f_ = self.model.add_parameters((self.attention_size, 2*self.hidden_size))
            self.Wa1_e_ = self.model.add_parameters((self.attention_size, self.hidden_size))
            self.wa2_ = self.model.add_parameters((1, self.attention_size))
        if self._attention.lower() == 'bilinear':
            self.Wa_ = self.model.add_parameters((2*self.hidden_size, self.hidden_size))
        
    def encode(self, src):        
        src_rev = list(reversed(src))
        # Bidirectional RNN or LSTM
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        src_rev = list(reversed(src))
        for (cw_l2r, cw_r2l) in zip(src, src_rev):
            l2r_state = l2r_state.add_input(dy.lookup(self.src_lookup, cw_l2r))
            r2l_state = r2l_state.add_input(dy.lookup(self.src_lookup, cw_r2l))
            l2r_contexts.append(l2r_state.output()) #[<s>, x_1, x_2, ..., </s>]
            r2l_contexts.append(r2l_state.output()) #[</s> x_n, x_{n-1}, ... <s>]
            
        encoded = dy.concatenate_cols([l2r_contexts[-1], r2l_contexts[-1]])
        r2l_contexts.reverse() #[<s>, x_1, x_2, ..., </s>]

        # Combine the left and right representations for every word
        h_fs = []  # foreign sentence hidden states
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        
        
        return h_fs, encoded
        
    def train(self, src_tgt, update=True):
        dy.renew_cg()
        W,b  = dy.parameter(self.W_), dy.parameter(self.b_)

        src, tgt = src_tgt
        # print('debug, len(src)=', len(src), 'len(tgt)=',len(tgt))
        h_fs, encoded = self.encode(src)
        Hf = dy.concatenate_cols(h_fs)   # refer to H^{(f)} in lecture notes
        
        if self._attention.lower() == 'mlp':
            Wa1_f = dy.parameter(self.Wa1_f_)
            Wa1_f_mult_Hf = Wa1_f * Hf # cache this result
        

        # Decoder
        if self.layer_depth == 1 and self.useEncoding:
            dec_state = self.dec_builder.initial_state(dy.transpose(encoded))
        else:
            dec_state = self.dec_builder.initial_state()
        
        # first, use the <s> symbol to predict the first token tgt[0]
        
        
        if self._attention.lower() == 'cheat':
            c_t = h_fs[0]
        else:
            c_t = dy.vecInput(2*self.hidden_size)
            
        cw = self.tgt_w2id['<s>']
        start = dy.concatenate([dy.lookup(self.tgt_lookup, cw ), c_t])
        dec_state = dec_state.add_input(start)
        output = dec_state.output()
        y_star = b + W*output 
        losses = [ dy.pickneglogsoftmax(y_star, tgt[0]) ]
        
        for idx, (cw, nw) in enumerate(zip(tgt, tgt[1:]+[self.tgt_w2id['</s>']])):
            h_e = dec_state.output()
            
            if idx == len(tgt)-1:  # predicting the </s> symbol
                c_t = dy.vecInput(2*self.hidden_size)
            else:
                if self._attention.lower() == 'cheat':
                    c_t = h_fs[idx+1]
                if self._attention.lower() == 'mlp':
                    c_t = self.__attentionScore_mlp(Hf, h_e, Wa1_f_mult_Hf=Wa1_f_mult_Hf)
                if self._attention.lower() == 'bilinear':
                    c_t = self.__attentionScore_bilinear(Hf, h_e)
                if self._attention.lower() == 'dot':
                    c_t = self.__attentionScore_dot(Hf, h_e)
                
            # Get the embedding for the current target word
            embed_t = dy.lookup(self.tgt_lookup, cw)
            # Create input vector to the decoder
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            output = dec_state.output()
            y_star = b + W*output 
            loss = dy.pickneglogsoftmax(y_star, nw) 
            losses.append(loss)
 
        loss = dy.esum(losses)    
        if update:
            loss.backward()
            self.trainer.update()
        return loss.scalar_value(), len(tgt)+1
    def train_batch(self, batch, update=True):
        dy.renew_cg()
        W,b  = dy.parameter(self.W_), dy.parameter(self.b_)

        src_batch = [x[0] for x in batch]
        tgt_batch = [x[1] for x in batch]
        batch_size = len(batch)
        # Encoder
        # src_batch  = [ [a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5], [c1,c2,c3,c4], ...]
        # transpose the batch into 
        #   src_cws: [[a1,b1,c1,..], [a2,b2,c2,..], ... [a5,b5,END,...]]
        #   src_len: [5,5,4,...]
        src_lens = [len(s) for s in src_batch]
        src_lmax = np.max(src_lens)
        src_cws = np.ones((src_lmax, len(batch)), dtype='uint') # use 1 to represent <s>
        
        for sid, s in enumerate(src_batch):
            src_cws[:src_lens[sid], sid] = s
            
        # Bidirectional RNN or LSTM
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for ii in range(src_lmax):
            cws_l2r, cws_r2l = src_cws[ii, :], src_cws[src_lmax-ii-1, :]
            l2r_state = l2r_state.add_input(dy.lookup_batch(self.src_lookup, cws_l2r))
            r2l_state = r2l_state.add_input(dy.lookup_batch(self.src_lookup, cws_r2l))
            l2r_contexts.append(l2r_state.output()) #[<s>, x_1, x_2, ..., </s>]
            r2l_contexts.append(r2l_state.output()) #[</s> x_n, x_{n-1}, ... <s>]

        r2l_contexts.reverse() #[<s>, x_1, x_2, ..., </s>]

        # Combine the left and right representations for every word
        h_fs = []  # foreign sentence hidden states
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        Hf = dy.concatenate_cols(h_fs)   # refer to H^{(f)} in lecture notes
        # Hf.dim() = 2hidden_size  * len(src) * batch_size
        if self._attention.lower() == 'mlp':
            Wa1_f = dy.parameter(self.Wa1_f_)
            Wa1_f_mult_Hf = Wa1_f * Hf # cache this result
        
        
        encoded = dy.concatenate_cols([l2r_contexts[-1], r2l_contexts[-1]])
        
        # Decoder
        tgt_lens = [len(s) for s in tgt_batch]
        tgt_lmax = np.max(tgt_lens)
        tgt_cws = np.ones((tgt_lmax+1, len(tgt_batch)), dtype='uint') * self.tgt_w2id['<s>']  
        masks = np.zeros((tgt_lmax+1, len(tgt_batch)), dtype='uint')
        tgt_cws[-1, :] = self.tgt_w2id['</s>']
        for sid, s in enumerate(tgt_batch):
            tgt_cws[tgt_lmax - len(s):-1, sid] = s
            masks[tgt_lmax - len(s):, sid] = 1
        
        dec_state = self.dec_builder.initial_state()
        
        if self._attention.lower() == 'cheat':
            #c_t = dy.select_cols(Hf, [0])
            c_t = dy.concatenate([ l2r_contexts[0], l2r_contexts[0]])
        else:
            c_t = dy.vecInput(2*self.hidden_size)
        cws = [self.tgt_w2id['<s>']]*batch_size
        embeddings = dy.lookup_batch(self.tgt_lookup, cws)   # dim: hidden_size * 1 * batch_size
        start = dy.concatenate([embeddings, c_t])   # c_t.dim: hidden_size * 1
        dec_state = dec_state.add_input(start)
        output = dec_state.output()
        y_star = b + W*output 
        losses = [ dy.pickneglogsoftmax_batch(y_star, tgt_cws[0]) ]
         
        for idx, (cws, nws) in enumerate(zip(tgt_cws[:-1], tgt_cws[1:])):
            h_e = dec_state.output()
            
            if idx == len(tgt_cws)-2:  # predicting the </s> symbol
                c_t = dy.vecInput(2*self.hidden_size)
                # c_t.dim() = 2hidden_size * 1 * 1
            else:
                
                if self._attention.lower() == 'cheat':
                    c_t = dy.concatenate([ l2r_contexts[idx+1], l2r_contexts[idx+1]])
                if self._attention.lower() == 'mlp':
                    c_t = self.__attentionScore_mlp(Hf, h_e, Wa1_f_mult_Hf=Wa1_f_mult_Hf)  
                    
                if self._attention.lower() == 'dot':
                    c_t = self.__attentionScore_dot(Hf, h_e)  
                if self._attention.lower() == 'bilinear':
                    c_t = self.__attentionScore_bilinear(Hf, h_e)  
                # c_t.dim() = 2hidden_size * 1 * batch_size
                
            # Get the embedding for the current target word
            embed_t = dy.lookup_batch(self.tgt_lookup, cws)
            # Create input vector to the decoder
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            output = dec_state.output()
            y_star = b + W*output 
            
            loss = dy.pickneglogsoftmax_batch(y_star, nws) 
            mask = masks[idx]
            mask_exp = dy.reshape(dy.inputVector(mask), (1,), batch_size)
            loss =  dy.cmult(loss, mask_exp)   
            losses.append(loss)
        
        loss = dy.sum_batches(dy.esum(losses))
        
        if update:
            loss.backward()
            self.trainer.update()
        
        return loss.scalar_value(), (tgt_lmax+1)*len(tgt_batch)  
    
    

    def __attentionScore_bilinear(self, Hf, he, argv={}):
        Wa = dy.parameter(self.Wa_)
        a_t = dy.transpose(Hf) * Wa * he
        return Hf * dy.softmax(a_t)
    def __attentionScore_dot(self, Hf, he, argv={}):
        a_t = dy.transpose(Hf)*dy.concatenate([he, he])
        return Hf * dy.softmax(a_t)

        # Calculates the context vector using a MLP
    # Hf: matrix of embeddings for the source words, size: (2*hidden, len(source))
    # h_e: hidden state of the decoder
    def __attentionScore_mlp(self, Hf, he, Wa1_f_mult_Hf=None):  # MLP 
        Wa1_e = dy.parameter(self.Wa1_e_)
        wa2 = dy.parameter(self.wa2_)
        
        if 'Wa1_f_mult_Hf' is None: 
            Wa1_f = dy.parameter(self.Wa1_f_)
            Wa1_f_mult_Hf = Wa1_f * Hf 
        # Hf dim(): 2*hidden_size  * len(src)
        # result is matrix of size (attention_size, len(src))
        
        #src_len = Hf.dim()[2]
        #Wa1_e_mult_repmat_he = Wa1_e * dy.concatenate_cols([he]*src_len)
        #   Wa1_e_mult_repmat_he is matrix of attention_size * len(src)
        # temp = Wa1_f_mult_Hf + Wa1_e_mult_repmat_he
        
        Wa1_e_mult_he = Wa1_e * he
        # result is matrix of attention_size * 1
        temp = dy.colwise_add(Wa1_f_mult_Hf, Wa1_e_mult_he)
        
        # Calculate the alignment score vector
        a_t = wa2 * dy.tanh(temp)   # size of temp: attention_size * len(src)
        alignment = dy.softmax(a_t)  # size: 1 * len(src)
        c_t = Hf * dy.transpose(alignment)
        
        return c_t
    def translate0(self, src, max_len=None):
        dy.renew_cg()
        W,b  = dy.parameter(self.W_), dy.parameter(self.b_)

        
        h_fs, encoded = self.encode(src)
        Hf = dy.concatenate_cols(h_fs)   # refer to H^{(f)} in lecture notes
        
        if self._attention.lower() == 'mlp':
            Wa1_f = dy.parameter(self.Wa1_f_)
            Wa1_f_mult_Hf = Wa1_f * Hf # cache this result
        
        # Decoder
        result = [self.tgt_w2id['<s>']]
        cw = result[-1]
        
        if self.layer_depth == 1 and self.useEncoding:
            dec_state = self.dec_builder.initial_state(dy.transpose(encoded))
        else:
            dec_state = self.dec_builder.initial_state()
            
        if max_len is None:
            max_len = int(1.1*len(src)+1)
            
        while len(result) < max_len:
            h_e = dec_state.output()
            if (len(result)==1):  # just start, no context now
                if self._attention.lower() == 'cheat':
                    c_t = h_fs[0]
                else:
                    c_t = dy.vecInput(self.hidden_size * 2)
            else:
                argv = {}
                if self._attention.lower() == 'cheat':
                    
                    if len(result)-1 < len(src):
                        c_t = h_fs[len(result)-1]
                    else:
                        c_t = dy.vecInput(self.hidden_size * 2)
                if self._attention.lower() == 'mlp':
                    c_t = self.__attentionScore_mlp(Hf, h_e, Wa1_f_mult_Hf=Wa1_f_mult_Hf)
                if self._attention.lower() == 'dot':
                    c_t = self.__attentionScore_dot(Hf, h_e)    
                if self._attention.lower() == 'bilinear':
                    c_t = self.__attentionScore_bilinear(Hf, h_e)   
                
            # Get the embedding for the current target word
            embed_t = dy.lookup(self.tgt_lookup, cw)
            # Create input vector to the decoder
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            output = dec_state.output()
            y_star = b + W*output 
            
            ydist = dy.softmax(y_star).vec_value()
            cw = np.argmax(ydist[2:]) +2 # [2:] and +2 are for avoiding the unk and start symbol
            if cw == [self.tgt_w2id['</s>']]:
                break
            result.append(cw)
        return result[1:]
    def translate(self, src, k=4, max_len=None):
        dy.renew_cg()
        W,b  = dy.parameter(self.W_), dy.parameter(self.b_)

        
        h_fs, encoded = self.encode(src)
        Hf = dy.concatenate_cols(h_fs)   # refer to H^{(f)} in lecture notes
        
        if self._attention.lower() == 'mlp':
            Wa1_f = dy.parameter(self.Wa1_f_)
            Wa1_f_mult_Hf = Wa1_f * Hf # cache this result
        
        # Decoder
        start_ = self.tgt_w2id['<s>']
        end_ = self.tgt_w2id['</s>']
        cws = {start_:(0,[])}   # 0: current total log probability; []: current sentence, 
        
        if self.layer_depth == 1 and self.useEncoding:
            dec_state = self.dec_builder.initial_state(dy.transpose(encoded))
        else:
            dec_state = self.dec_builder.initial_state()
        pool = [(0,[start_], dec_state)]   # 0: current total log probability; [start_]: current sentence, dec_state: not including the last word in current sentence
        
        max_len = int(1.1*len(src)+1)  if max_len is None else max_len
            
        for pos in range(max_len):
            newPoolCandidates = []
            for oldLogProb, oldSent, oldState in pool:
                cw = oldSent[-1]
                if cw == end_:
                    newPoolCandidates.append((oldLogProb, oldSent, oldState))
                    continue
                h_e = oldState.output()
                if (pos==0):  # just start, no context now
                    if self._attention.lower() == 'cheat':
                        c_t = h_fs[0]
                    else:
                        c_t = dy.vecInput(self.hidden_size * 2)
                else:
                    argv = {}
                    if self._attention.lower() == 'cheat':

                        if len(result)-1 < len(src):
                            c_t = h_fs[len(result)-1]
                        else:
                            c_t = dy.vecInput(self.hidden_size * 2)
                    if self._attention.lower() == 'mlp':
                        c_t = self.__attentionScore_mlp(Hf, h_e, Wa1_f_mult_Hf=Wa1_f_mult_Hf)
                    if self._attention.lower() == 'dot':
                        c_t = self.__attentionScore_dot(Hf, h_e)    
                    if self._attention.lower() == 'bilinear':
                        c_t = self.__attentionScore_bilinear(Hf, h_e) 

                    
                    
                # Get the embedding for the current target word
                embed_t = dy.lookup(self.tgt_lookup, cw)
                # Create input vector to the decoder
                x_t = dy.concatenate([embed_t, c_t])
                newState = oldState.add_input(x_t)
                output = newState.output()
                y_star = b + W*output 

                ydist = dy.softmax(y_star).vec_value()
                bestk = np.argsort(ydist[2:])[-k:] +2 # [2:] and +2 are for avoiding the unk symbol and start symbol 
                for nw in bestk:
                    logProb = np.log(ydist[nw])
                    newPoolCandidates.append((oldLogProb+logProb, oldSent+[nw], newState))
                    
            
            newPoolCandidates.sort(reverse=True)  # logProb high to low
            pool = newPoolCandidates[:k]
        #print('finally len(pool)=', len(pool) )
        result = pool[0][1]
        return result[1:-1] if result[-1]==end_ else result[1:]
    
    
def makeToyVocab(vocabSize=100, corpusSizes=[2000,1000],  keepOrder=True):
    # use corpusSizes=[2000,1000] to get a traning corpus of size 2000 and a validation of size 1000

    vocab1 = ['f'+str(ii+1) for ii in range(vocabSize)]
    vocab2 = ['e'+str(ii+1) for ii in range(vocabSize)]
    wordFreq = np.random.rand(vocabSize)
    wordFreq = wordFreq / sum(wordFreq)

    corpus = []
    for corpusSize in corpusSizes:
        toy1, toy2 = [], []
        for ii in range(corpusSize):
            sentLen = int(5+20*np.random.rand(1))
            wordCts = np.random.multinomial(sentLen, wordFreq)
            src, tgt = [], []
            for jj, ct in enumerate(wordCts):
                if ct>0:
                    src.extend([vocab1[jj]]*ct)
                    tgt.extend([vocab2[jj]]*ct)
            if keepOrder:
                tmp = list(zip(src, tgt))
                np.random.shuffle(tmp)
                src, tgt = zip(*tmp)
            else:
                np.random.shuffle(src)
                np.random.shuffle(tgt)
            toy1.append(src)
            toy2.append(tgt)
        corpus.append((toy1, toy2))
    return corpus



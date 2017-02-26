import sys
#sys.argv.extend('--dynet-gpu-ids 2 --dynet-mem 8000'.split())
print(sys.argv)
import dynet as dy
#print(dy.__version__)

from collections import Counter, defaultdict
import numpy as np
import time, pickle

from nltk.translate import bleu_score
from mt7 import AttentionTranslator, makeToyVocab

_embedding_size=512
_hidden_size=512
_attention_size=512
_layer_depth=1
_attention='dot'
_cell='lstm'
_minWordCount = 5
_num_epoch = 30
_batch_size = 32
#_trainer='adam'
_dropout=None
_useToyData = False

prefix='exp1_'
trainDisplayInterval = 20
validDisplayInterval = 200


def mylog(logFile, msg):
    print(msg)
    logFile.write(msg+'\n')

if __name__ == '__main__': 
    tic = str(time.time()).split('.')[0]
    _para = {'tic':tic,'_embedding_size':_embedding_size, '_hidden_size':_hidden_size, '_attention_size':_attention_size, \
             '_layer_depth':_layer_depth, '_attention':_attention, '_minWordCount':_minWordCount, '_cell':_cell, \
            '_num_epoch':_num_epoch, '_batch_size':_batch_size, '_useToyData':_useToyData, '_dropout':_dropout}
    with open(prefix+_attention+'_'+tic+'.para', 'wb') as f:
        pickle.dump(_para, f)
    #log = open('null.log', 'w')
    logfile = prefix+_attention+'_'+tic+'.log'
    log = open(logfile, 'w')
    mylog(log, "using log file:"+logfile)
        
        
    maxTrain = None
    maxValid = None

    if _useToyData:
        train, valid = makeToyVocab(vocabSize = 50, corpusSizes=[32001,500], keepOrder=True)
        trainde, trainen = train
        validde, validen = valid 
    else:
        with open('data/en-de/train.en-de.low.filt.de', 'r') as file:
            lines = file.readlines()
            trainde = [[w.lower() for w in l.strip().split()] for l in lines[:maxTrain]]
        with open('data/en-de/train.en-de.low.filt.en', 'r') as file:
            lines = file.readlines()
            trainen = [[w.lower() for w in l.strip().split()] for l in lines[:maxTrain]]
        assert(len(trainen)==len(trainde))

        with open('data/en-de/valid.en-de.low.de', 'r') as file:
            lines = file.readlines()
            validde =  [[w.lower() for w in l.strip().split()] for l in lines]
        with open('data/en-de/valid.en-de.low.en', 'r') as file:
            lines = file.readlines()
            validen =  [[w.lower() for w in l.strip().split()] for l in lines]
        assert(len(validen)==len(validde))
    
    
    cten = Counter()
    for l in trainen:
        cten.update(l)
    ctde = Counter()
    for l in trainde:
        ctde.update(l)
        

    w2idde=defaultdict(lambda:0)
    w2idde["<unk>"] = 0
    w2idde["<s>"] = 1
    w2idde["</s>"] = 2
    id2wde=defaultdict(lambda:"<unk>")
    id2wde[0] = "<unk>"
    id2wde[1] = "<s>"
    id2wde[2] = "</s>"

    w2iden, id2wen = w2idde.copy(), id2wde.copy()

    for word, freq in ctde.items():
        if freq >= _minWordCount:
            wid = len(w2idde)
            id2wde[wid] = word
            w2idde[word] = wid
    for word, freq in cten.items():
        if freq >= _minWordCount:
            wid = len(w2iden)
            id2wen[wid] = word
            w2iden[word] = wid 
    mylog(log, "len(ctde)=%d, len(id2wde)=%d, len(w2idde)=%d" %(len(ctde),len(id2wde),len(w2idde)))
    mylog(log, "len(cten)=%d, len(id2wen)=%d, len(w2iden)=%d" %(len(cten),len(id2wen),len(w2iden)))
    #print(len(ctde), len(id2wde), len(w2idde))
    #print(len(cten), len(id2wen), len(w2iden))
  


    # train wihtout minibatching
    builder = dy.LSTMBuilder if _cell.lower()=='lstm' else dy.SimpleRNNBuilder
    translator = AttentionTranslator((w2idde, id2wde), (w2iden, id2wen), embedding_size=_embedding_size, \
                                     hidden_size=_hidden_size, attention_size=_attention_size, \
                                     layer_depth=_layer_depth, builder=builder,  \
                                     attention=_attention, dropout=_dropout)
    
    translator.trainer = dy.AdamTrainer(translator.model)
    

    
    
    #unknownPenalty = 0   # positive value: reduct <unk> in output
    #smoothParameter = 1.0 # 0.0: completed random output, valid perplexity ~ 3800

    validSampleIndex = [0,1,2,3,4]

    train_de_id = [[w2idde[w] for w in sent] for sent in trainde]
    train_en_id = [[w2iden[w] for w in sent] for sent in trainen]
    valid_de_id = [[w2idde[w] for w in sent] for sent in validde]
    valid_en_id = [[w2iden[w] for w in sent] for sent in validen]

    trainLossHistory, validLossHistory=[], []
    mylog(log, 'num_epochs=%d\n'%_num_epoch)
    
    if _batch_size is None:  # no minibatching
            
        for epoch in range(_num_epoch):
            mylog(log, 'epoch = '+str(epoch))
            if epoch ==10:
                translator.trainer = dy.SimpleSGDTrainer(translator.model)
            trainWc = trainLoss = 0

            for ii in range(len(train_de_id)):
                src_tgt = train_de_id[ii], train_en_id[ii]
                loss, wc = translator.train(src_tgt)
                trainWc += wc
                trainLoss += loss #.scalar_value()
                if ii%trainDisplayInterval==trainDisplayInterval-1 and trainWc>0: 
                    translator.trainer.status()
                    #print('  training index #', ii, ', loss=', trainLoss / trainWc)
                    mylog(log, '  training index # %d, perplexity loss=%.5f' % (ii, np.exp( trainLoss / trainWc)))
                    trainWc = trainLoss = 0
                    trainLossHistory.append(trainLoss)

                if ii%validDisplayInterval==validDisplayInterval-1  or ii==len(train_de_id)-1:  # check validation  loss/perplexity 
                    validResults = []
                    reference = []
                    for jj in range(len( valid_de_id)): 
                        src = valid_de_id[jj]
                        ref = validen[jj]  # validen contains words, while valid_en_id contains ids
                        tgt = translator.translate(src)
                        validResults.append([ id2wen[wid] for wid in tgt])
                        reference.append([ref])
                    bleu = bleu_score.corpus_bleu(reference, validResults)
                    #print('    training index #', ii, ', BLEU on validation=', bleu)
                    mylog(log, '    training index # %d, BLEU on validation=%.5f' % (ii, bleu))
                    validLossHistory.append(bleu)


                    # check example translation
                    mylog(log, "    Translation examples")
                    for idx in validSampleIndex:
                        mylog(log, '      hypothesis: ' + (" ".join(validResults[idx]).strip())+'')
                        mylog(log, '      reference:  ' + (" ".join(reference[idx][0]).strip())+'')
            translator.trainer.update_epoch(1.0)
            translator.model.save(_attention+'_'+tic+'_epoch'+str(epoch+1)+'.model')
        mylog(log, 'training done')
        mylog(log, 'parameters: '+_para)
        log.close()
    else:  # minibatching
        train = zip(train_de_id, train_en_id)
        valid = zip(valid_de_id, valid_en_id)
        train = sorted(train, key=lambda x: -len(x[0]))
        # train_order = [0, 32, 64, ...]
        train_order = [x*_batch_size for x in range((len(train)-1) // _batch_size)]
        
        mylog(log, 'total number of batches: %d' %len(train_order))
        for epoch in range(_num_epoch):
            mylog(log, 'epoch = '+str(epoch))
            
            if epoch ==5:
                translator.trainer = dy.SimpleSGDTrainer(translator.model)
                
            np.random.shuffle(train_order) 
            trainWc = trainLoss = 0

            for ii, tidx in enumerate( train_order):
                batch = train[tidx:tidx+_batch_size]

                loss,batchWc = translator.train_batch(batch)

                trainWc += batchWc
                trainLoss += loss #.scalar_value()
                trainLossHistory.append(loss) #.scalar_value())

                if ii%trainDisplayInterval==trainDisplayInterval-1 and trainWc>0:  # check training loss/perplexity
                    translator.trainer.status()
                    mylog(log, '  training batch # %d, perplexity loss=%.5f' % (ii, np.exp(trainLoss / trainWc)))
                    trainWc = trainLoss = 0

                if ii%validDisplayInterval==validDisplayInterval-1 or ii==len(train_order)-1:  # check validation BLEU score
                    validResults = []
                    validResults0 = []
                    reference = []
                    for jj in range(len( valid_de_id)): 
                        src = valid_de_id[jj]
                        ref = validen[jj]  # validen contains words, while valid_en contains ids
                        tgt = translator.translate(src)
                        #tgt0 =  translator.translate0(src)
                        validResults.append([ id2wen[wid] for wid in tgt])
                        #validResults0.append([ id2wen[wid] for wid in tgt0])
                        reference.append([ref])
                    bleu = bleu_score.corpus_bleu(reference, validResults)
                    mylog(log, '    training batch # %d, BLEU on validation=%.5f' % (ii, bleu))
         #           bleu = bleu_score.corpus_bleu(reference, validResults0)
         #           mylog(log, '    training batch # %d, BLEU on validation=%.5f (w/o beam search)' % (ii, bleu))
                    validLossHistory.append(bleu)

                    # check example translation
                    mylog(log, "    Translation examples")
                    for idx in validSampleIndex:
                        mylog(log, '      hypothesis: ' + (" ".join(validResults[idx]).strip())+'')
                        #mylog(log, '      hypothesis0:' + (" ".join(validResults0[idx]).strip())+'')
                        mylog(log, '      reference:  ' + (" ".join(reference[idx][0]).strip())+'')
                        mylog(log, '----------------------------------------------------------')
            translator.trainer.update_epoch(1.0)
            translator.model.save(prefix+_attention+'_'+tic+'_epoch'+str(epoch+1)+'.model')
        mylog(log, 'training done')
        mylog(log, 'parameters: '+str(_para))
        log.close()
    

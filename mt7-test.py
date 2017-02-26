import sys
# sys.argv.extend('--dynet-gpu-ids 4 --dynet-mem 5000'.split())
print(sys.argv)
import dynet as dy
#print(dy.__version__)

from collections import Counter, defaultdict
import numpy as np
import time, pickle

from nltk.translate import bleu_score
from mt7 import AttentionTranslator

if __name__ == '__main__': 

    modelFile = sys.argv[1]
    if modelFile.split('.')[1] != 'model':
        print('argument error. example: python mt7-test.py bilinear_1487794222_epoch27.model')
        sys.exit(-1)
    configFile = modelFile.split('_epoch')[0]+'.para'
    testOutputFile = modelFile.split('_epoch')[0]+'.test'
    blindOutputFile = modelFile.split('_epoch')[0]+'.blind'
    
    
    k = 4
    with open(configFile, 'rb') as f:
        para = pickle.load(f)
        
     
    maxTrain, maxValid, maxTest, maxBlind = None, None, None, None

    with open('data/en-de/train.en-de.low.filt.de', 'r') as file:
        lines = file.readlines()
        trainde = [[w.lower() for w in l.strip().split()] for l in lines[:maxTrain]]
    with open('data/en-de/train.en-de.low.filt.en', 'r') as file:
        lines = file.readlines()
        trainen = [[w.lower() for w in l.strip().split()] for l in lines[:maxTrain]]
    assert(len(trainen)==len(trainde))

    with open('data/en-de/valid.en-de.low.de', 'r') as file:
        lines = file.readlines()
        validde =  [[w.lower() for w in l.strip().split()] for l in lines[:maxValid]]
    with open('data/en-de/valid.en-de.low.en', 'r') as file:
        lines = file.readlines()
        validen =  [[w.lower() for w in l.strip().split()] for l in lines[:maxValid]]
    assert(len(validde)==len(validen))
    
    with open('data/en-de/test.en-de.low.de', 'r') as file:
        lines = file.readlines()
        testde =  [[w.lower() for w in l.strip().split()] for l in lines[:maxTest]]
    with open('data/en-de/test.en-de.low.en', 'r') as file:
        lines = file.readlines()
        testen =  [[w.lower() for w in l.strip().split()] for l in lines[:maxTest]]
    assert(len(testde)==len(testen))

    with open('data/en-de/blind.en-de.low.de', 'r') as file:
        lines = file.readlines()
        blindde =  [[w.lower() for w in l.strip().split()] for l in lines[:maxBlind]]

    
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
        if freq >= para['_minWordCount']:
            wid = len(w2idde)
            id2wde[wid] = word
            w2idde[word] = wid
    for word, freq in cten.items():
        if freq >= para['_minWordCount']:
            wid = len(w2iden)
            id2wen[wid] = word
            w2iden[word] = wid        
    print(len(ctde), len(id2wde), len(w2idde))
    print(len(cten), len(id2wen), len(w2iden))
    


        # train wihtout minibatching
    builder = dy.LSTMBuilder if para['_cell'].lower()=='lstm' else dy.SimpleRNNBuilder

    translator2 = AttentionTranslator((w2idde, id2wde), (w2iden, id2wen), embedding_size=para['_embedding_size'],hidden_size=para['_hidden_size'], attention_size=para['_attention_size'],layer_depth=para['_layer_depth'], builder=builder, attention=para['_attention'])
    print('using model file: ', modelFile)
    translator2.model.load(modelFile)
    print('model file loaded')
    
    printIndex = [0,1,2,3,4]

    train_de_id = [[w2idde[w] for w in sent] for sent in trainde]
    train_en_id = [[w2iden[w] for w in sent] for sent in trainen]
    valid_de_id = [[w2idde[w] for w in sent] for sent in validde]
    valid_en_id = [[w2iden[w] for w in sent] for sent in validen]
    test_de_id = [[w2idde[w] for w in sent] for sent in testde]
    test_en_id = [[w2iden[w] for w in sent] for sent in testen]
    blind_de_id = [[w2idde[w] for w in sent] for sent in blindde]
    
    print('evaluating on valid data')
    testResults, testResults0, reference = [], [], []
    for jj in range(len( valid_de_id)): 
        src = valid_de_id[jj]
        ref = validen[jj]  
        tgt = translator2.translate(src, k=k)
        tgt0 = translator2.translate0(src)
        testResults.append([ id2wen[wid] for wid in tgt])
        testResults0.append([ id2wen[wid] for wid in tgt0])
        reference.append([ref])
    bleu = bleu_score.corpus_bleu(reference, testResults)
    print(', BLEU on valid, =', bleu)
    bleu = bleu_score.corpus_bleu(reference, testResults0)
    print(', BLEU on valid, w/o beam search=', bleu)


    
    print('evaluating on test data')
    testResults, testResults0, reference = [], [], []
    for jj in range(len( test_de_id)): 
        src = test_de_id[jj]
        ref = testen[jj]  
        tgt = translator2.translate(src, k=k)
        tgt0 = translator2.translate0(src)
        testResults.append([ id2wen[wid] for wid in tgt])
        testResults0.append([ id2wen[wid] for wid in tgt0])
        reference.append([ref])
    bleu = bleu_score.corpus_bleu(reference, testResults)
    print(', BLEU on test, =', bleu)
    bleu = bleu_score.corpus_bleu(reference, testResults0)
    print(', BLEU on test, w/o beam search=', bleu)

    print('translation examples')
    for idx in printIndex:
        print('         source:  ', " ".join(trainde[idx]))
        print('      hypothesis: ', " ".join(testResults[idx]).strip())
        print('      hypothesis0:', " ".join(testResults0[idx]).strip())
        print('      reference:  ', " ".join(reference[idx][0]).strip())
        print('-------------------------------')
    with open(testOutputFile, 'w') as f:
        for res in testResults:
            f.write(" ".join(res).strip() + '\n')            
            
    print('translating on blind data')
    testResults, testResults0, reference = [], [], []
    for jj in range(len( blind_de_id)): 
        src = blind_de_id[jj]
        tgt = translator2.translate(src, k=k)
        #tgt0 = translator2.translate0(src)
        testResults.append([ id2wen[wid] for wid in tgt])
        #testResults0.append([ id2wen[wid] for wid in tgt0])

    with open(blindOutputFile, 'w') as f:
        for res in testResults:
            f.write(" ".join(res).strip() + '\n')
    print('test done, k=',k)

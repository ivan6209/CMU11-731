To train, run:

python mt7-train.py --dyent-gpu-ids 0 --dynet-mem 8000

Prior to running mt7-train.py, all parameters should be set inside mt7-train.py  (from line 14 to 30)

During training, the script would output:

prefix_dot_1487794184_epoch29.model

prefix_dot_1487794184.log

prefix_dot_1487794184.para

Here ''prefix'' is the pre-defined prefix (line 27 of mt7-train.py), ''dot'' is the attention score function (dot product, line 18 of mt7-train.py), ''1487794184'' is the start time.

The ''.para'' file stores the training parameters and can be loaded with pickle.


To test, run:

python mt7-test.py prefix_dot_1487794184_epoch29.model --dyent-gpu-ids 0 --dynet-mem 8000

where prefix_dot_1487794184_epoch29.model is the model generated during training.

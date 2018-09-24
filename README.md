 This code is for ACL2018 paper "Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism"
## Environment
 - python2.7
 - [requirements.txt](https://github.com/xiangrongzeng/copy_re/blob/master/requirements.txt)

## Data

You need to modify the data path in const.py before running the code.
The pre-processed data is released.

WebNLG:

 - [dataset](https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj)
 - [pre-trained word embedding](https://drive.google.com/open?id=1LOT2-JxjjglCFyxv-JQAJlJvEmleSXZl)

NYT:

 - [dataset](https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3)
 - [pre-trained word embedding](https://drive.google.com/open?id=1yVjN-0lZid6YJmsX5g8x_YKiCfnRy8IL)
 


## Train

 python main.py -c config.json -t 0 -cell lstm

The cell can be "gru" or "lstm".
You can specify the GPU card number by "-g". For exampe, "python main.py -c config.json -t 0 -cell lstm -g 0".

## Test or Valid

You need to set the epochs of the model in main.py first. Then run the following commands:
 python main.py -c config.json -t 1 -cell lstm

"-t 1" means test and "-t 2" means valid.



# Results for MNIST dataset

## 1 hidden layer of 16 neuron He intialized, elu activation function, cross entropy loss, nadam optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.24601953
Iteration 10: loss = 0.061340604
Iteration 20: loss = 0.04186616
Iteration 30: loss = 0.035088763
Iteration 40: loss = 0.03133954
Iteration 50: loss = 0.02880185
Iteration 60: loss = 0.026939582
Iteration 70: loss = 0.025421605
Iteration 80: loss = 0.024120415
Iteration 90: loss = 0.022980707
Iteration 100: loss = 0.021966562
Iteration 110: loss = 0.021057272
Iteration 120: loss = 0.020238798
Iteration 130: loss = 0.019499602
Iteration 140: loss = 0.018829359
Iteration 150: loss = 0.018218907

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1

Confusion Matrix:
[[964, 0, 1, 2, 1, 5, 4, 1, 2, 0],
 [0, 1115, 4, 1, 0, 1, 3, 2, 9, 0],
 [6, 7, 966, 12, 8, 1, 7, 5, 20, 0],
 [3, 1, 12, 941, 0, 19, 0, 13, 18, 3],
 [1, 1, 4, 1, 939, 0, 9, 2, 3, 22],
 [5, 2, 1, 21, 9, 822, 13, 0, 13, 6],
 [9, 3, 3, 0, 6, 14, 919, 0, 4, 0],
 [1, 10, 24, 3, 5, 1, 0, 953, 2, 29],
 [10, 3, 5, 20, 8, 14, 8, 11, 890, 5],
 [8, 4, 2, 11, 27, 8, 0, 11, 4, 934]]

Class 0: TP = 964, FP = 43, FN = 16, TN = 8977
Precision = 0.95729893
Recall(Sensitivity) = 0.98367345
F1-score: = 0.970307
---------------------------------------------
Class 1: TP = 1115, FP = 31, FN = 20, TN = 8834
Precision = 0.9729494
Recall(Sensitivity) = 0.98237884
F1-score: = 0.97764134
---------------------------------------------
Class 2: TP = 966, FP = 56, FN = 66, TN = 8912
Precision = 0.94520545
Recall(Sensitivity) = 0.93604654
F1-score: = 0.9406037
---------------------------------------------
Class 3: TP = 941, FP = 71, FN = 69, TN = 8919
Precision = 0.9298419
Recall(Sensitivity) = 0.9316832
F1-score: = 0.93076164
---------------------------------------------
Class 4: TP = 939, FP = 64, FN = 43, TN = 8954
Precision = 0.93619144
Recall(Sensitivity) = 0.9562118
F1-score: = 0.94609576
---------------------------------------------
Class 5: TP = 822, FP = 63, FN = 70, TN = 9045
Precision = 0.9288136
Recall(Sensitivity) = 0.92152464
F1-score: = 0.92515475
---------------------------------------------
Class 6: TP = 919, FP = 44, FN = 39, TN = 8998
Precision = 0.95430946
Recall(Sensitivity) = 0.9592902
F1-score: = 0.9567933
---------------------------------------------
Class 7: TP = 953, FP = 45, FN = 75, TN = 8927
Precision = 0.9549098
Recall(Sensitivity) = 0.9270428
F1-score: = 0.94077
---------------------------------------------
Class 8: TP = 890, FP = 75, FN = 84, TN = 8951
Precision = 0.9222798
Recall(Sensitivity) = 0.9137577
F1-score: = 0.91799897
---------------------------------------------
Class 9: TP = 934, FP = 65, FN = 75, TN = 8926
Precision = 0.9349349
Recall(Sensitivity) = 0.92566895
F1-score: = 0.9302789
---------------------------------------------
Loss: 0.019970542
Accuracy: 94.43%
```


## 1 hidden layer of 16 neuron, leaky relu activation function, cross entropy loss, momentum optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.3874951
Iteration 1: loss = 0.35273188
Iteration 2: loss = 0.30907303
Iteration 3: loss = 0.27532813
Iteration 4: loss = 0.25414827
Iteration 5: loss = 0.24122891
Iteration 6: loss = 0.23262021
Iteration 7: loss = 0.22634895
Iteration 8: loss = 0.22146404
Iteration 9: loss = 0.217356
Iteration 10: loss = 0.21364972
Iteration 11: loss = 0.21010962
Iteration 12: loss = 0.2065748
Iteration 13: loss = 0.20294635
Iteration 14: loss = 0.19916221
Iteration 15: loss = 0.19519605
Iteration 16: loss = 0.19105382
Iteration 17: loss = 0.18676251
Iteration 18: loss = 0.18237115
Iteration 19: loss = 0.17793275
Iteration 20: loss = 0.1735014
Iteration 21: loss = 0.16910349
Iteration 22: loss = 0.164758
Iteration 23: loss = 0.16046302
Iteration 24: loss = 0.15618855
Iteration 25: loss = 0.15189737
Iteration 26: loss = 0.14757995
Iteration 27: loss = 0.143256
Iteration 28: loss = 0.13896568
Iteration 29: loss = 0.13475055
Iteration 30: loss = 0.13069105
Iteration 31: loss = 0.12684299
Iteration 32: loss = 0.123252295
Iteration 33: loss = 0.11993448
Iteration 34: loss = 0.11686909
Iteration 35: loss = 0.11401367
Iteration 36: loss = 0.11130052
Iteration 37: loss = 0.1086603
Iteration 38: loss = 0.10606339
Iteration 39: loss = 0.10351627
Iteration 40: loss = 0.10105577
Iteration 41: loss = 0.09872074
Iteration 42: loss = 0.09654221
Iteration 43: loss = 0.094520934
Iteration 44: loss = 0.09263696
Iteration 45: loss = 0.09087283
Iteration 46: loss = 0.08921462
Iteration 47: loss = 0.08765488
Iteration 48: loss = 0.08617678
Iteration 49: loss = 0.0847621
Iteration 50: loss = 0.08339318
Iteration 51: loss = 0.08205769
Iteration 52: loss = 0.08075711
Iteration 53: loss = 0.0795022
Iteration 54: loss = 0.07830273
Iteration 55: loss = 0.07716525
Iteration 56: loss = 0.07609162
Iteration 57: loss = 0.075077444
Iteration 58: loss = 0.07411044
Iteration 59: loss = 0.07317344
Iteration 60: loss = 0.072251305
Iteration 61: loss = 0.07133816
Iteration 62: loss = 0.07044052
Iteration 63: loss = 0.06957208
Iteration 64: loss = 0.06874836
Iteration 65: loss = 0.06797542
Iteration 66: loss = 0.06724837
Iteration 67: loss = 0.0665535
Iteration 68: loss = 0.06587683
Iteration 69: loss = 0.06521059
Iteration 70: loss = 0.064555824
Iteration 71: loss = 0.063920505
Iteration 72: loss = 0.06331242
Iteration 73: loss = 0.062733755
Iteration 74: loss = 0.062181458
Iteration 75: loss = 0.06164879
Iteration 76: loss = 0.061130695
Iteration 77: loss = 0.060626213
Iteration 78: loss = 0.060136218
Iteration 79: loss = 0.05966192
Iteration 80: loss = 0.05920431
Iteration 81: loss = 0.05876224
Iteration 82: loss = 0.05833336
Iteration 83: loss = 0.057916153
Iteration 84: loss = 0.05750999
Iteration 85: loss = 0.05711543
Iteration 86: loss = 0.05673259
Iteration 87: loss = 0.05636114
Iteration 88: loss = 0.05600004
Iteration 89: loss = 0.055648778
Iteration 90: loss = 0.055306587
Iteration 91: loss = 0.054973256
Iteration 92: loss = 0.054648373
Iteration 93: loss = 0.05433153
Iteration 94: loss = 0.054022554
Iteration 95: loss = 0.05372057
Iteration 96: loss = 0.0534253
Iteration 97: loss = 0.05313657
Iteration 98: loss = 0.052854527
Iteration 99: loss = 0.0525793
Iteration 100: loss = 0.052310627
Iteration 101: loss = 0.052048262
Iteration 102: loss = 0.051791325
Iteration 103: loss = 0.05153956
Iteration 104: loss = 0.051292863
Iteration 105: loss = 0.051051185
Iteration 106: loss = 0.05081446
Iteration 107: loss = 0.050582755
Iteration 108: loss = 0.05035585
Iteration 109: loss = 0.05013356
Iteration 110: loss = 0.049915865
Iteration 111: loss = 0.049702242
Iteration 112: loss = 0.049492598
Iteration 113: loss = 0.049287055
Iteration 114: loss = 0.049085345
Iteration 115: loss = 0.048887488
Iteration 116: loss = 0.04869323
Iteration 117: loss = 0.04850244
Iteration 118: loss = 0.048315126
Iteration 119: loss = 0.04813107
Iteration 120: loss = 0.047950335
Iteration 121: loss = 0.047772735
Iteration 122: loss = 0.047598336
Iteration 123: loss = 0.04742698
Iteration 124: loss = 0.047258355
Iteration 125: loss = 0.04709248
Iteration 126: loss = 0.046929326
Iteration 127: loss = 0.046768565
Iteration 128: loss = 0.04661035
Iteration 129: loss = 0.046454668
Iteration 130: loss = 0.04630125
Iteration 131: loss = 0.046150174
Iteration 132: loss = 0.046001367
Iteration 133: loss = 0.045854803
Iteration 134: loss = 0.04571034
Iteration 135: loss = 0.045568027
Iteration 136: loss = 0.04542771
Iteration 137: loss = 0.045289516
Iteration 138: loss = 0.04515334
Iteration 139: loss = 0.04501911
Iteration 140: loss = 0.04488665
Iteration 141: loss = 0.044756074
Iteration 142: loss = 0.044627223
Iteration 143: loss = 0.044500045
Iteration 144: loss = 0.0443746
Iteration 145: loss = 0.044250835
Iteration 146: loss = 0.0441287
Iteration 147: loss = 0.044008248
Iteration 148: loss = 0.0438893
Iteration 149: loss = 0.043771915
Iteration 150: loss = 0.04365603

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.042081427
Accuracy: 87.29%
```

## 1 hidden layer of 16 neuron, elu activation function, cross entropy loss, rmsprop optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.34491262
Iteration 1: loss = 0.31871104
Iteration 2: loss = 0.21395057
Iteration 3: loss = 0.1898906
Iteration 4: loss = 0.1652369
Iteration 5: loss = 0.14896278
Iteration 6: loss = 0.132481
Iteration 7: loss = 0.12166284
Iteration 8: loss = 0.1133716
Iteration 9: loss = 0.1064149
Iteration 10: loss = 0.10052747
Iteration 11: loss = 0.09568267
Iteration 12: loss = 0.091630824
Iteration 13: loss = 0.08825354
Iteration 14: loss = 0.084751345
Iteration 15: loss = 0.08150187
Iteration 16: loss = 0.07784914
Iteration 17: loss = 0.074750654
Iteration 18: loss = 0.071869835
Iteration 19: loss = 0.06947952
Iteration 20: loss = 0.06755884
Iteration 21: loss = 0.06564622
Iteration 22: loss = 0.06436811
Iteration 23: loss = 0.062473673
Iteration 24: loss = 0.061228145
Iteration 25: loss = 0.05915435
Iteration 26: loss = 0.05763888
Iteration 27: loss = 0.055770937
Iteration 28: loss = 0.054383047
Iteration 29: loss = 0.052892447
Iteration 30: loss = 0.05181316
Iteration 31: loss = 0.05058968
Iteration 32: loss = 0.04980161
Iteration 33: loss = 0.048744213
Iteration 34: loss = 0.048228998
Iteration 35: loss = 0.047240168
Iteration 36: loss = 0.047037546
Iteration 37: loss = 0.046194106
Iteration 38: loss = 0.04708903
Iteration 39: loss = 0.048214998
Iteration 40: loss = 0.05168631
Iteration 41: loss = 0.05073785
Iteration 42: loss = 0.048866417
Iteration 43: loss = 0.044913124
Iteration 44: loss = 0.042739168
Iteration 45: loss = 0.04094778
Iteration 46: loss = 0.039686833
Iteration 47: loss = 0.038803533
Iteration 48: loss = 0.038127944
Iteration 49: loss = 0.037692
Iteration 50: loss = 0.037369683
Iteration 51: loss = 0.037320253
Iteration 52: loss = 0.03740467
Iteration 53: loss = 0.037902743
Iteration 54: loss = 0.038319387
Iteration 55: loss = 0.03889904
Iteration 56: loss = 0.03854665
Iteration 57: loss = 0.037978295
Iteration 58: loss = 0.037009213
Iteration 59: loss = 0.03656146
Iteration 60: loss = 0.036939748
Iteration 61: loss = 0.03808432
Iteration 62: loss = 0.038652405
Iteration 63: loss = 0.037674222
Iteration 64: loss = 0.035381805
Iteration 65: loss = 0.03318178
Iteration 66: loss = 0.032088634
Iteration 67: loss = 0.03136917
Iteration 68: loss = 0.031172916
Iteration 69: loss = 0.031209825
Iteration 70: loss = 0.03187487
Iteration 71: loss = 0.032858353
Iteration 72: loss = 0.03360653
Iteration 73: loss = 0.03334652
Iteration 74: loss = 0.032630708
Iteration 75: loss = 0.031955574
Iteration 76: loss = 0.032190397
Iteration 77: loss = 0.032299127
Iteration 78: loss = 0.03294026
Iteration 79: loss = 0.032437142
Iteration 80: loss = 0.031589124
Iteration 81: loss = 0.031643365
Iteration 82: loss = 0.031878397
Iteration 83: loss = 0.031132506
Iteration 84: loss = 0.030239655
Iteration 85: loss = 0.029090527
Iteration 86: loss = 0.028066289
Iteration 87: loss = 0.027535954
Iteration 88: loss = 0.027087316
Iteration 89: loss = 0.026915468
Iteration 90: loss = 0.026757425
Iteration 91: loss = 0.026857127
Iteration 92: loss = 0.027005393
Iteration 93: loss = 0.027590012
Iteration 94: loss = 0.028526634
Iteration 95: loss = 0.030304752
Iteration 96: loss = 0.03263565
Iteration 97: loss = 0.032464694
Iteration 98: loss = 0.029937409
Iteration 99: loss = 0.028403139
Iteration 100: loss = 0.027343063
Iteration 101: loss = 0.02605759
Iteration 102: loss = 0.025858624
Iteration 103: loss = 0.026090983
Iteration 104: loss = 0.027426168
Iteration 105: loss = 0.029640283
Iteration 106: loss = 0.031180391
Iteration 107: loss = 0.028262721
Iteration 108: loss = 0.026327318
Iteration 109: loss = 0.0245037
Iteration 110: loss = 0.023905516
Iteration 111: loss = 0.023579637
Iteration 112: loss = 0.023408616
Iteration 113: loss = 0.023321586
Iteration 114: loss = 0.023375267
Iteration 115: loss = 0.02373331
Iteration 116: loss = 0.025013292
Iteration 117: loss = 0.027843783
Iteration 118: loss = 0.03236839
Iteration 119: loss = 0.030082164
Iteration 120: loss = 0.026888765
Iteration 121: loss = 0.023801548
Iteration 122: loss = 0.023055645
Iteration 123: loss = 0.02307042
Iteration 124: loss = 0.023497136
Iteration 125: loss = 0.024207467
Iteration 126: loss = 0.024566306
Iteration 127: loss = 0.025008386
Iteration 128: loss = 0.025355458
Iteration 129: loss = 0.02559946
Iteration 130: loss = 0.026502468
Iteration 131: loss = 0.025430698
Iteration 132: loss = 0.024516849
Iteration 133: loss = 0.023137012
Iteration 134: loss = 0.022329718
Iteration 135: loss = 0.021965813
Iteration 136: loss = 0.021745143
Iteration 137: loss = 0.021820879
Iteration 138: loss = 0.022047963
Iteration 139: loss = 0.02287318
Iteration 140: loss = 0.024057403
Iteration 141: loss = 0.025751092
Iteration 142: loss = 0.026113149
Iteration 143: loss = 0.025752012
Iteration 144: loss = 0.024843689
Iteration 145: loss = 0.023081442
Iteration 146: loss = 0.022918284
Iteration 147: loss = 0.022434773
Iteration 148: loss = 0.02222973
Iteration 149: loss = 0.021673692
Iteration 150: loss = 0.021294897

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.021050507
Accuracy: 93.72%
```

## 1 hidden layer of 16 neuron, selu activation function, cross entropy loss, rmsprop optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.37218064
Iteration 1: loss = 0.2705778
Iteration 2: loss = 0.24489057
Iteration 3: loss = 0.19107726
Iteration 4: loss = 0.16661336
Iteration 5: loss = 0.13974607
Iteration 6: loss = 0.12555034
Iteration 7: loss = 0.116013125
Iteration 8: loss = 0.10968092
Iteration 9: loss = 0.102018625
Iteration 10: loss = 0.09590408
Iteration 11: loss = 0.087790504
Iteration 12: loss = 0.082811125
Iteration 13: loss = 0.07802339
Iteration 14: loss = 0.07486366
Iteration 15: loss = 0.07190146
Iteration 16: loss = 0.070913136
Iteration 17: loss = 0.07209563
Iteration 18: loss = 0.07312242
Iteration 19: loss = 0.0694489
Iteration 20: loss = 0.06624952
Iteration 21: loss = 0.06189239
Iteration 22: loss = 0.05891078
Iteration 23: loss = 0.056565482
Iteration 24: loss = 0.05457567
Iteration 25: loss = 0.05307922
Iteration 26: loss = 0.051827468
Iteration 27: loss = 0.050768483
Iteration 28: loss = 0.050065994
Iteration 29: loss = 0.049277525
Iteration 30: loss = 0.049363133
Iteration 31: loss = 0.049831938
Iteration 32: loss = 0.0516618
Iteration 33: loss = 0.050908003
Iteration 34: loss = 0.04755446
Iteration 35: loss = 0.044837154
Iteration 36: loss = 0.04334651
Iteration 37: loss = 0.043048706
Iteration 38: loss = 0.043731708
Iteration 39: loss = 0.045497477
Iteration 40: loss = 0.045119744
Iteration 41: loss = 0.044758704
Iteration 42: loss = 0.041737463
Iteration 43: loss = 0.040807948
Iteration 44: loss = 0.040245254
Iteration 45: loss = 0.04120438
Iteration 46: loss = 0.04204941
Iteration 47: loss = 0.04431458
Iteration 48: loss = 0.042902723
Iteration 49: loss = 0.040919825
Iteration 50: loss = 0.038395952
Iteration 51: loss = 0.036522288
Iteration 52: loss = 0.035453346
Iteration 53: loss = 0.034704424
Iteration 54: loss = 0.034328446
Iteration 55: loss = 0.034149654
Iteration 56: loss = 0.03435754
Iteration 57: loss = 0.03521506
Iteration 58: loss = 0.037008688
Iteration 59: loss = 0.038420126
Iteration 60: loss = 0.038024355
Iteration 61: loss = 0.03549323
Iteration 62: loss = 0.033446632
Iteration 63: loss = 0.032924134
Iteration 64: loss = 0.033231862
Iteration 65: loss = 0.035184346
Iteration 66: loss = 0.034862895
Iteration 67: loss = 0.034980405
Iteration 68: loss = 0.032432683
Iteration 69: loss = 0.031684395
Iteration 70: loss = 0.03120625
Iteration 71: loss = 0.031183021
Iteration 72: loss = 0.031815402
Iteration 73: loss = 0.032309506
Iteration 74: loss = 0.0331747
Iteration 75: loss = 0.03251119
Iteration 76: loss = 0.031857837
Iteration 77: loss = 0.03126449
Iteration 78: loss = 0.03172646
Iteration 79: loss = 0.033086218
Iteration 80: loss = 0.034373887
Iteration 81: loss = 0.03374802
Iteration 82: loss = 0.03128459
Iteration 83: loss = 0.029216953
Iteration 84: loss = 0.027985511
Iteration 85: loss = 0.027615683
Iteration 86: loss = 0.027387012
Iteration 87: loss = 0.027333409
Iteration 88: loss = 0.027285568
Iteration 89: loss = 0.02740169
Iteration 90: loss = 0.02746003
Iteration 91: loss = 0.027838157
Iteration 92: loss = 0.028127871
Iteration 93: loss = 0.028867869
Iteration 94: loss = 0.029571824
Iteration 95: loss = 0.030435277
Iteration 96: loss = 0.030530462
Iteration 97: loss = 0.030665798
Iteration 98: loss = 0.03073336
Iteration 99: loss = 0.032109067
Iteration 100: loss = 0.031207351
Iteration 101: loss = 0.028570266
Iteration 102: loss = 0.027466934
Iteration 103: loss = 0.026533086
Iteration 104: loss = 0.025764015
Iteration 105: loss = 0.025351143
Iteration 106: loss = 0.025036868
Iteration 107: loss = 0.025015566
Iteration 108: loss = 0.02514512
Iteration 109: loss = 0.025717048
Iteration 110: loss = 0.026727919
Iteration 111: loss = 0.028305853
Iteration 112: loss = 0.028191986
Iteration 113: loss = 0.027514778
Iteration 114: loss = 0.025467534
Iteration 115: loss = 0.024619669
Iteration 116: loss = 0.024177995
Iteration 117: loss = 0.02416656
Iteration 118: loss = 0.024766184
Iteration 119: loss = 0.025524925
Iteration 120: loss = 0.027958132
Iteration 121: loss = 0.029320091
Iteration 122: loss = 0.029293632
Iteration 123: loss = 0.027943604
Iteration 124: loss = 0.02443936
Iteration 125: loss = 0.0234437
Iteration 126: loss = 0.02299629
Iteration 127: loss = 0.023208462
Iteration 128: loss = 0.023804625
Iteration 129: loss = 0.025406517
Iteration 130: loss = 0.026505033
Iteration 131: loss = 0.027230753
Iteration 132: loss = 0.026776694
Iteration 133: loss = 0.025374755
Iteration 134: loss = 0.024361636
Iteration 135: loss = 0.023645516
Iteration 136: loss = 0.022995394
Iteration 137: loss = 0.022623898
Iteration 138: loss = 0.02225445
Iteration 139: loss = 0.022209046
Iteration 140: loss = 0.022248121
Iteration 141: loss = 0.022785235
Iteration 142: loss = 0.02356146
Iteration 143: loss = 0.024210753
Iteration 144: loss = 0.024200797
Iteration 145: loss = 0.023022491
Iteration 146: loss = 0.022476451
Iteration 147: loss = 0.0223993
Iteration 148: loss = 0.023953944
Iteration 149: loss = 0.026737956
Iteration 150: loss = 0.02991341

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.026730014
Accuracy: 91.59%
```

## 1 hidden layer of 16 neuron, elu activation function, cross entropy loss, adam optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.3719147
Iteration 1: loss = 0.2964634
Iteration 2: loss = 0.25842255
Iteration 3: loss = 0.23274226
Iteration 4: loss = 0.21297446
Iteration 5: loss = 0.19649708
Iteration 6: loss = 0.18232957
Iteration 7: loss = 0.17026755
Iteration 8: loss = 0.16005565
Iteration 9: loss = 0.15117204
Iteration 10: loss = 0.1429934
Iteration 11: loss = 0.13503757
Iteration 12: loss = 0.12713668
Iteration 13: loss = 0.11945283
Iteration 14: loss = 0.112320885
Iteration 15: loss = 0.10598394
Iteration 16: loss = 0.100450575
Iteration 17: loss = 0.0955654
Iteration 18: loss = 0.09120693
Iteration 19: loss = 0.08733287
Iteration 20: loss = 0.08386269
Iteration 21: loss = 0.08063583
Iteration 22: loss = 0.07752291
Iteration 23: loss = 0.07453012
Iteration 24: loss = 0.07175827
Iteration 25: loss = 0.06928965
Iteration 26: loss = 0.067113
Iteration 27: loss = 0.06513685
Iteration 28: loss = 0.063265935
Iteration 29: loss = 0.061461054
Iteration 30: loss = 0.059733853
Iteration 31: loss = 0.058105826
Iteration 32: loss = 0.0565875
Iteration 33: loss = 0.055186093
Iteration 34: loss = 0.053909764
Iteration 35: loss = 0.05275358
Iteration 36: loss = 0.051689807
Iteration 37: loss = 0.050680824
Iteration 38: loss = 0.049702495
Iteration 39: loss = 0.048756722
Iteration 40: loss = 0.04786321
Iteration 41: loss = 0.047040936
Iteration 42: loss = 0.046293713
Iteration 43: loss = 0.045608047
Iteration 44: loss = 0.0449633
Iteration 45: loss = 0.044343937
Iteration 46: loss = 0.043745473
Iteration 47: loss = 0.043171197
Iteration 48: loss = 0.04262536
Iteration 49: loss = 0.042108186
Iteration 50: loss = 0.04161721
Iteration 51: loss = 0.041150324
Iteration 52: loss = 0.040706087
Iteration 53: loss = 0.040280998
Iteration 54: loss = 0.039869666
Iteration 55: loss = 0.039468024
Iteration 56: loss = 0.039076608
Iteration 57: loss = 0.038698573
Iteration 58: loss = 0.038335945
Iteration 59: loss = 0.037987586
Iteration 60: loss = 0.037651505
Iteration 61: loss = 0.03732737
Iteration 62: loss = 0.03701558
Iteration 63: loss = 0.036715366
Iteration 64: loss = 0.03642435
Iteration 65: loss = 0.03614079
Iteration 66: loss = 0.0358647
Iteration 67: loss = 0.035597038
Iteration 68: loss = 0.03533799
Iteration 69: loss = 0.035086785
Iteration 70: loss = 0.034842666
Iteration 71: loss = 0.03460552
Iteration 72: loss = 0.034375392
Iteration 73: loss = 0.03415158
Iteration 74: loss = 0.03393332
Iteration 75: loss = 0.03372008
Iteration 76: loss = 0.033512067
Iteration 77: loss = 0.033309817
Iteration 78: loss = 0.033113226
Iteration 79: loss = 0.032921515
Iteration 80: loss = 0.03273367
Iteration 81: loss = 0.03254924
Iteration 82: loss = 0.032368157
Iteration 83: loss = 0.03219064
Iteration 84: loss = 0.032016717
Iteration 85: loss = 0.031846303
Iteration 86: loss = 0.03167945
Iteration 87: loss = 0.03151595
Iteration 88: loss = 0.031355366
Iteration 89: loss = 0.031197373
Iteration 90: loss = 0.03104199
Iteration 91: loss = 0.03088938
Iteration 92: loss = 0.030739564
Iteration 93: loss = 0.030592473
Iteration 94: loss = 0.030447891
Iteration 95: loss = 0.030305728
Iteration 96: loss = 0.030165832
Iteration 97: loss = 0.030028086
Iteration 98: loss = 0.029892493
Iteration 99: loss = 0.029758966
Iteration 100: loss = 0.029627483
Iteration 101: loss = 0.029497994
Iteration 102: loss = 0.029370328
Iteration 103: loss = 0.029244414
Iteration 104: loss = 0.029120136
Iteration 105: loss = 0.02899748
Iteration 106: loss = 0.028876387
Iteration 107: loss = 0.028756823
Iteration 108: loss = 0.028638698
Iteration 109: loss = 0.028522022
Iteration 110: loss = 0.028406763
Iteration 111: loss = 0.028292852
Iteration 112: loss = 0.028180242
Iteration 113: loss = 0.028068952
Iteration 114: loss = 0.027958855
Iteration 115: loss = 0.027850058
Iteration 116: loss = 0.027742393
Iteration 117: loss = 0.027635956
Iteration 118: loss = 0.02753067
Iteration 119: loss = 0.027426537
Iteration 120: loss = 0.027323494
Iteration 121: loss = 0.027221505
Iteration 122: loss = 0.02712056
Iteration 123: loss = 0.027020661
Iteration 124: loss = 0.02692175
Iteration 125: loss = 0.02682383
Iteration 126: loss = 0.02672689
Iteration 127: loss = 0.0266309
Iteration 128: loss = 0.026535869
Iteration 129: loss = 0.026441721
Iteration 130: loss = 0.026348494
Iteration 131: loss = 0.026256166
Iteration 132: loss = 0.026164746
Iteration 133: loss = 0.026074203
Iteration 134: loss = 0.025984552
Iteration 135: loss = 0.025895787
Iteration 136: loss = 0.025807923
Iteration 137: loss = 0.025720881
Iteration 138: loss = 0.025634708
Iteration 139: loss = 0.02554934
Iteration 140: loss = 0.025464816
Iteration 141: loss = 0.025381083
Iteration 142: loss = 0.025298137
Iteration 143: loss = 0.025215996
Iteration 144: loss = 0.02513465
Iteration 145: loss = 0.025054079
Iteration 146: loss = 0.02497431
Iteration 147: loss = 0.024895348
Iteration 148: loss = 0.024817139
Iteration 149: loss = 0.024739727
Iteration 150: loss = 0.02466298

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.025647987
Accuracy: 92.46%
```

## 1 hidden layer of 16 neuron, elu activation function, cross entropy loss, nadam optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.3536123
Iteration 1: loss = 0.25432098
Iteration 2: loss = 0.2161119
Iteration 3: loss = 0.1905043
Iteration 4: loss = 0.17048356
Iteration 5: loss = 0.15493268
Iteration 6: loss = 0.14214265
Iteration 7: loss = 0.1310357
Iteration 8: loss = 0.12166914
Iteration 9: loss = 0.11374203
Iteration 10: loss = 0.106925756
Iteration 11: loss = 0.10094485
Iteration 12: loss = 0.09556855
Iteration 13: loss = 0.090636745
Iteration 14: loss = 0.08607475
Iteration 15: loss = 0.08185815
Iteration 16: loss = 0.07797542
Iteration 17: loss = 0.07440663
Iteration 18: loss = 0.07112327
Iteration 19: loss = 0.06809822
Iteration 20: loss = 0.065309584
Iteration 21: loss = 0.06274288
Iteration 22: loss = 0.060386952
Iteration 23: loss = 0.058229968
Iteration 24: loss = 0.056259252
Iteration 25: loss = 0.054460794
Iteration 26: loss = 0.052820247
Iteration 27: loss = 0.051323432
Iteration 28: loss = 0.049956173
Iteration 29: loss = 0.04870549
Iteration 30: loss = 0.0475589
Iteration 31: loss = 0.04650477
Iteration 32: loss = 0.045532215
Iteration 33: loss = 0.044630915
Iteration 34: loss = 0.043792207
Iteration 35: loss = 0.043008953
Iteration 36: loss = 0.042275015
Iteration 37: loss = 0.041585453
Iteration 38: loss = 0.040936038
Iteration 39: loss = 0.040322904
Iteration 40: loss = 0.039742526
Iteration 41: loss = 0.03919179
Iteration 42: loss = 0.03866792
Iteration 43: loss = 0.038168278
Iteration 44: loss = 0.03769073
Iteration 45: loss = 0.037233207
Iteration 46: loss = 0.036794234
Iteration 47: loss = 0.036372554
Iteration 48: loss = 0.03596708
Iteration 49: loss = 0.035576966
Iteration 50: loss = 0.035201367
Iteration 51: loss = 0.03483957
Iteration 52: loss = 0.03449101
Iteration 53: loss = 0.034155037
Iteration 54: loss = 0.03383105
Iteration 55: loss = 0.033518516
Iteration 56: loss = 0.033217046
Iteration 57: loss = 0.032926086
Iteration 58: loss = 0.032645203
Iteration 59: loss = 0.032373738
Iteration 60: loss = 0.032111336
Iteration 61: loss = 0.031857476
Iteration 62: loss = 0.0316116
Iteration 63: loss = 0.031373255
Iteration 64: loss = 0.031142008
Iteration 65: loss = 0.03091737
Iteration 66: loss = 0.030698968
Iteration 67: loss = 0.030486407
Iteration 68: loss = 0.030279355
Iteration 69: loss = 0.030077526
Iteration 70: loss = 0.02988056
Iteration 71: loss = 0.029688219
Iteration 72: loss = 0.029500235
Iteration 73: loss = 0.029316498
Iteration 74: loss = 0.029136745
Iteration 75: loss = 0.028960872
Iteration 76: loss = 0.028788762
Iteration 77: loss = 0.028620286
Iteration 78: loss = 0.028455293
Iteration 79: loss = 0.0282937
Iteration 80: loss = 0.028135462
Iteration 81: loss = 0.02798042
Iteration 82: loss = 0.027828464
Iteration 83: loss = 0.027679538
Iteration 84: loss = 0.027533477
Iteration 85: loss = 0.027390199
Iteration 86: loss = 0.02724961
Iteration 87: loss = 0.027111584
Iteration 88: loss = 0.026976114
Iteration 89: loss = 0.026842989
Iteration 90: loss = 0.02671216
Iteration 91: loss = 0.026583575
Iteration 92: loss = 0.026457164
Iteration 93: loss = 0.026332863
Iteration 94: loss = 0.026210576
Iteration 95: loss = 0.026090242
Iteration 96: loss = 0.025971783
Iteration 97: loss = 0.025855185
Iteration 98: loss = 0.02574041
Iteration 99: loss = 0.02562737
Iteration 100: loss = 0.025516039
Iteration 101: loss = 0.025406348
Iteration 102: loss = 0.02529826
Iteration 103: loss = 0.02519174
Iteration 104: loss = 0.025086751
Iteration 105: loss = 0.024983248
Iteration 106: loss = 0.02488121
Iteration 107: loss = 0.024780575
Iteration 108: loss = 0.024681339
Iteration 109: loss = 0.02458345
Iteration 110: loss = 0.024486884
Iteration 111: loss = 0.024391618
Iteration 112: loss = 0.0242976
Iteration 113: loss = 0.024204819
Iteration 114: loss = 0.024113255
Iteration 115: loss = 0.024022896
Iteration 116: loss = 0.023933688
Iteration 117: loss = 0.023845585
Iteration 118: loss = 0.023758624
Iteration 119: loss = 0.023672737
Iteration 120: loss = 0.023587894
Iteration 121: loss = 0.023504095
Iteration 122: loss = 0.023421306
Iteration 123: loss = 0.023339497
Iteration 124: loss = 0.023258682
Iteration 125: loss = 0.02317883
Iteration 126: loss = 0.023099907
Iteration 127: loss = 0.023021862
Iteration 128: loss = 0.022944694
Iteration 129: loss = 0.022868391
Iteration 130: loss = 0.022792917
Iteration 131: loss = 0.02271823
Iteration 132: loss = 0.02264436
Iteration 133: loss = 0.022571236
Iteration 134: loss = 0.022498913
Iteration 135: loss = 0.022427306
Iteration 136: loss = 0.022356449
Iteration 137: loss = 0.022286294
Iteration 138: loss = 0.022216834
Iteration 139: loss = 0.022148093
Iteration 140: loss = 0.022080027
Iteration 141: loss = 0.022012588
Iteration 142: loss = 0.0219458
Iteration 143: loss = 0.021879653
Iteration 144: loss = 0.021814125
Iteration 145: loss = 0.021749219
Iteration 146: loss = 0.021684887
Iteration 147: loss = 0.021621143
Iteration 148: loss = 0.02155798
Iteration 149: loss = 0.02149536
Iteration 150: loss = 0.021433324

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.02411928
Accuracy: 92.94%
```

## 1 hidden layer of 16 neuron He intialized, elu activation function, cross entropy loss, nadam optimizer, 0.01 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.24853164
Iteration 1: loss = 0.17384005
Iteration 2: loss = 0.172105
Iteration 3: loss = 0.12794612
Iteration 4: loss = 0.10475904
Iteration 5: loss = 0.09161225
Iteration 6: loss = 0.08313847
Iteration 7: loss = 0.07661963
Iteration 8: loss = 0.07120152
Iteration 9: loss = 0.06680047
Iteration 10: loss = 0.063227445
Iteration 11: loss = 0.060253914
Iteration 12: loss = 0.057717063
Iteration 13: loss = 0.05550614
Iteration 14: loss = 0.053538453
Iteration 15: loss = 0.051761333
Iteration 16: loss = 0.050149046
Iteration 17: loss = 0.04868931
Iteration 18: loss = 0.047369804
Iteration 19: loss = 0.046172056
Iteration 20: loss = 0.045074515
Iteration 21: loss = 0.0440589
Iteration 22: loss = 0.043113224
Iteration 23: loss = 0.042230695
Iteration 24: loss = 0.041406754
Iteration 25: loss = 0.040637005
Iteration 26: loss = 0.03991697
Iteration 27: loss = 0.03924198
Iteration 28: loss = 0.038607426
Iteration 29: loss = 0.038009427
Iteration 30: loss = 0.037444573
Iteration 31: loss = 0.036909986
Iteration 32: loss = 0.036403242
Iteration 33: loss = 0.03592219
Iteration 34: loss = 0.035464823
Iteration 35: loss = 0.03502938
Iteration 36: loss = 0.0346143
Iteration 37: loss = 0.034218263
Iteration 38: loss = 0.033839963
Iteration 39: loss = 0.03347857
Iteration 40: loss = 0.03313323
Iteration 41: loss = 0.03280315
Iteration 42: loss = 0.0324877
Iteration 43: loss = 0.032186024
Iteration 44: loss = 0.03189746
Iteration 45: loss = 0.03162135
Iteration 46: loss = 0.03135694
Iteration 47: loss = 0.031103496
Iteration 48: loss = 0.030860186
Iteration 49: loss = 0.030626355
Iteration 50: loss = 0.030401217
Iteration 51: loss = 0.030184144
Iteration 52: loss = 0.029974453
Iteration 53: loss = 0.029771706
Iteration 54: loss = 0.029575273
Iteration 55: loss = 0.029384732
Iteration 56: loss = 0.02919961
Iteration 57: loss = 0.02901953
Iteration 58: loss = 0.028844154
Iteration 59: loss = 0.028673174
Iteration 60: loss = 0.028506309
Iteration 61: loss = 0.02834333
Iteration 62: loss = 0.028184047
Iteration 63: loss = 0.02802826
Iteration 64: loss = 0.02787582
Iteration 65: loss = 0.027726566
Iteration 66: loss = 0.027580371
Iteration 67: loss = 0.027437165
Iteration 68: loss = 0.027296739
Iteration 69: loss = 0.027159039
Iteration 70: loss = 0.027023913
Iteration 71: loss = 0.026891325
Iteration 72: loss = 0.026761077
Iteration 73: loss = 0.026633134
Iteration 74: loss = 0.026507318
Iteration 75: loss = 0.026383545
Iteration 76: loss = 0.026261762
Iteration 77: loss = 0.026141774
Iteration 78: loss = 0.02602353
Iteration 79: loss = 0.025906917
Iteration 80: loss = 0.02579181
Iteration 81: loss = 0.025678167
Iteration 82: loss = 0.025565892
Iteration 83: loss = 0.025454873
Iteration 84: loss = 0.025345065
Iteration 85: loss = 0.025236374
Iteration 86: loss = 0.025128704
Iteration 87: loss = 0.02502202
Iteration 88: loss = 0.024916286
Iteration 89: loss = 0.02481138
Iteration 90: loss = 0.024707295
Iteration 91: loss = 0.024604008
Iteration 92: loss = 0.024501497
Iteration 93: loss = 0.024399703
Iteration 94: loss = 0.024298588
Iteration 95: loss = 0.024198115
Iteration 96: loss = 0.024098229
Iteration 97: loss = 0.023998896
Iteration 98: loss = 0.02390009
Iteration 99: loss = 0.023801792
Iteration 100: loss = 0.02370399
Iteration 101: loss = 0.02360672
Iteration 102: loss = 0.023509944
Iteration 103: loss = 0.023413716
Iteration 104: loss = 0.023318013
Iteration 105: loss = 0.023222761
Iteration 106: loss = 0.023127982
Iteration 107: loss = 0.023033615
Iteration 108: loss = 0.022939675
Iteration 109: loss = 0.022846188
Iteration 110: loss = 0.022753194
Iteration 111: loss = 0.022660641
Iteration 112: loss = 0.022568544
Iteration 113: loss = 0.022476848
Iteration 114: loss = 0.022385573
Iteration 115: loss = 0.022294763
Iteration 116: loss = 0.022204395
Iteration 117: loss = 0.02211445
Iteration 118: loss = 0.022024984
Iteration 119: loss = 0.021935973
Iteration 120: loss = 0.021847436
Iteration 121: loss = 0.02175936
Iteration 122: loss = 0.021671802
Iteration 123: loss = 0.021584736
Iteration 124: loss = 0.021498224
Iteration 125: loss = 0.02141217
Iteration 126: loss = 0.021326672
Iteration 127: loss = 0.021241646
Iteration 128: loss = 0.021157123
Iteration 129: loss = 0.021073103
Iteration 130: loss = 0.020989612
Iteration 131: loss = 0.020906644
Iteration 132: loss = 0.020824203
Iteration 133: loss = 0.020742312
Iteration 134: loss = 0.020660993
Iteration 135: loss = 0.020580191
Iteration 136: loss = 0.020499928
Iteration 137: loss = 0.020420235
Iteration 138: loss = 0.020341113
Iteration 139: loss = 0.020262541
Iteration 140: loss = 0.020184515
Iteration 141: loss = 0.020107059
Iteration 142: loss = 0.020030199
Iteration 143: loss = 0.019953884
Iteration 144: loss = 0.019878138
Iteration 145: loss = 0.019802952
Iteration 146: loss = 0.019728344
Iteration 147: loss = 0.01965432
Iteration 148: loss = 0.0195809
Iteration 149: loss = 0.01950806
Iteration 150: loss = 0.019435808

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.020764414
Accuracy: 94%
```

## 1 hidden layer of 16 neuron He intialized, elu activation function, cross entropy loss, rmsprop optimizer, 0.001 learning rate, 150 iterations
```shell
Training dataset dimensions
DATA: 60000, 784
LABELS: 60000, 1

Iteration 0: loss = 0.24677078
Iteration 1: loss = 0.2388423
Iteration 2: loss = 0.21298032
Iteration 3: loss = 0.15988456
Iteration 4: loss = 0.14047888
Iteration 5: loss = 0.1261486
Iteration 6: loss = 0.118391745
Iteration 7: loss = 0.1130256
Iteration 8: loss = 0.10622413
Iteration 9: loss = 0.09997055
Iteration 10: loss = 0.09290307
Iteration 11: loss = 0.087909035
Iteration 12: loss = 0.08333954
Iteration 13: loss = 0.07978126
Iteration 14: loss = 0.076613806
Iteration 15: loss = 0.07398029
Iteration 16: loss = 0.071871854
Iteration 17: loss = 0.06971319
Iteration 18: loss = 0.06834289
Iteration 19: loss = 0.06585617
Iteration 20: loss = 0.06456384
Iteration 21: loss = 0.06167645
Iteration 22: loss = 0.060261145
Iteration 23: loss = 0.05775044
Iteration 24: loss = 0.056446746
Iteration 25: loss = 0.054524943
Iteration 26: loss = 0.05345076
Iteration 27: loss = 0.051977213
Iteration 28: loss = 0.051202755
Iteration 29: loss = 0.05012474
Iteration 30: loss = 0.049887355
Iteration 31: loss = 0.049607083
Iteration 32: loss = 0.050465155
Iteration 33: loss = 0.051421408
Iteration 34: loss = 0.051232077
Iteration 35: loss = 0.04852791
Iteration 36: loss = 0.0467846
Iteration 37: loss = 0.044679567
Iteration 38: loss = 0.043954935
Iteration 39: loss = 0.042980034
Iteration 40: loss = 0.04269745
Iteration 41: loss = 0.042364188
Iteration 42: loss = 0.04221212
Iteration 43: loss = 0.04206813
Iteration 44: loss = 0.04128655
Iteration 45: loss = 0.04087681
Iteration 46: loss = 0.039714955
Iteration 47: loss = 0.039456926
Iteration 48: loss = 0.038881578
Iteration 49: loss = 0.03929435
Iteration 50: loss = 0.03996599
Iteration 51: loss = 0.040486176
Iteration 52: loss = 0.040678672
Iteration 53: loss = 0.03875852
Iteration 54: loss = 0.03754323
Iteration 55: loss = 0.03646411
Iteration 56: loss = 0.035950147
Iteration 57: loss = 0.035972342
Iteration 58: loss = 0.035400085
Iteration 59: loss = 0.03527222
Iteration 60: loss = 0.034654923
Iteration 61: loss = 0.034870867
Iteration 62: loss = 0.0351493
Iteration 63: loss = 0.03568002
Iteration 64: loss = 0.03572677
Iteration 65: loss = 0.034637116
Iteration 66: loss = 0.0337328
Iteration 67: loss = 0.03263511
Iteration 68: loss = 0.03205942
Iteration 69: loss = 0.031580705
Iteration 70: loss = 0.03135666
Iteration 71: loss = 0.031294968
Iteration 72: loss = 0.031671196
Iteration 73: loss = 0.03282279
Iteration 74: loss = 0.03570899
Iteration 75: loss = 0.039184302
Iteration 76: loss = 0.037714764
Iteration 77: loss = 0.03333973
Iteration 78: loss = 0.030853054
Iteration 79: loss = 0.02981294
Iteration 80: loss = 0.029544726
Iteration 81: loss = 0.029432734
Iteration 82: loss = 0.029929597
Iteration 83: loss = 0.030408239
Iteration 84: loss = 0.03156659
Iteration 85: loss = 0.030870043
Iteration 86: loss = 0.031231713
Iteration 87: loss = 0.030334929
Iteration 88: loss = 0.030120121
Iteration 89: loss = 0.02983188
Iteration 90: loss = 0.029372597
Iteration 91: loss = 0.029327057
Iteration 92: loss = 0.028853241
Iteration 93: loss = 0.028855652
Iteration 94: loss = 0.028507812
Iteration 95: loss = 0.028584229
Iteration 96: loss = 0.028485771
Iteration 97: loss = 0.02880084
Iteration 98: loss = 0.02909625
Iteration 99: loss = 0.02984736
Iteration 100: loss = 0.029840957
Iteration 101: loss = 0.030094037
Iteration 102: loss = 0.029712455
Iteration 103: loss = 0.029996088
Iteration 104: loss = 0.030602803
Iteration 105: loss = 0.029229792
Iteration 106: loss = 0.028710742
Iteration 107: loss = 0.028037738
Iteration 108: loss = 0.027284408
Iteration 109: loss = 0.027182767
Iteration 110: loss = 0.026738023
Iteration 111: loss = 0.026955673
Iteration 112: loss = 0.026792971
Iteration 113: loss = 0.027027762
Iteration 114: loss = 0.026790423
Iteration 115: loss = 0.026705785
Iteration 116: loss = 0.026386688
Iteration 117: loss = 0.026177172
Iteration 118: loss = 0.025951754
Iteration 119: loss = 0.025844388
Iteration 120: loss = 0.025826946
Iteration 121: loss = 0.026012575
Iteration 122: loss = 0.026471134
Iteration 123: loss = 0.027095733
Iteration 124: loss = 0.027348066
Iteration 125: loss = 0.026938193
Iteration 126: loss = 0.02636339
Iteration 127: loss = 0.026898062
Iteration 128: loss = 0.028310116
Iteration 129: loss = 0.029214818
Iteration 130: loss = 0.027798584
Iteration 131: loss = 0.026970465
Iteration 132: loss = 0.026220879
Iteration 133: loss = 0.025000963
Iteration 134: loss = 0.024672035
Iteration 135: loss = 0.024184026
Iteration 136: loss = 0.024102826
Iteration 137: loss = 0.023910534
Iteration 138: loss = 0.024032533
Iteration 139: loss = 0.02409475
Iteration 140: loss = 0.024892965
Iteration 141: loss = 0.025632503
Iteration 142: loss = 0.02663953
Iteration 143: loss = 0.0258481
Iteration 144: loss = 0.024860239
Iteration 145: loss = 0.024061156
Iteration 146: loss = 0.023776708
Iteration 147: loss = 0.023710286
Iteration 148: loss = 0.023876294
Iteration 149: loss = 0.024028277
Iteration 150: loss = 0.02420069

Testing dataset dimensions
DATA: 10000, 784
LABELS: 10000, 1
Loss: 0.024805205
Accuracy: 92.659996%
```
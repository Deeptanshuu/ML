## Experiment - 6 Support Vector Machine for Binary Classification

## Outputs
```
=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances          74               74      %
Incorrectly Classified Instances        26               26      %
Kappa statistic                          0.3948
Mean absolute error                      0.26  
Root mean squared error                  0.5099
Relative absolute error                 57.722  %
Root relative squared error            107.5058 %
Total Number of Instances              100     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.848    0.471    0.778      0.848    0.812      0.399    0.689     0.760     c0
                 0.529    0.152    0.643      0.529    0.581      0.399    0.689     0.500     c1
Weighted Avg.    0.740    0.362    0.732      0.740    0.733      0.399    0.689     0.672     

=== Confusion Matrix ===

  a  b   <-- classified as
 56 10 |  a = c0
 16 18 |  b = c1
```

![OUTPUT](https://github.com/deeptanshuu/ML/raw/main/exp-6/op.png)
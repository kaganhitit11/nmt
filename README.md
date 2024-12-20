# Neural Machine Translation with Attention

COMP442 Spring 2024 HW2 - Neural Machine Translation
Oğuz Kağan Hitit, ID: 0076757

Training and Testing Results:

In order to see the results in a Google Sheets: https://docs.google.com/spreadsheets/d/1MvUTeSCTFmDT6LkpAjv_bqpHFMYXhYoaqcReFbON5zk/edit?usp=sharing

Results printed below and the results in the Google Sheets is the same.

Without Attention:

LSTM without attention tranining
Epoch 1: loss=4.998241463419547, BLEU=0.0011479642080724055
Epoch 2: loss=3.996688697723325, BLEU=0.0017062427961482374
Epoch 3: loss=3.511165390368191, BLEU=0.00316063203279716
Epoch 4: loss=3.1449371612567423, BLEU=0.002413275363736793
Epoch 5: loss=2.8450206156834397, BLEU=0.002314480291287329
LSTM without attention testing
—beam-width 4: The average BLEU score over the test set was 0.0033150283131815335
—greedy: The average BLEU score over the test set was 0.005382159705579552

GRU without attention tranining
Epoch 1: loss=4.269332038232098, BLEU=0.006900321559142036
Epoch 2: loss=3.1098769443147334, BLEU=0.004882407691189685
Epoch 3: loss=2.6096889786333204, BLEU=0.005811108405691903
Epoch 4: loss=2.280318247513883, BLEU=0.005252005468375502
Epoch 5: loss=2.0579686177187644, BLEU=0.004506624349737127
GRU without attention testing
—beam-width 4: The average BLEU score over the test set was 0.006512001051013772
--greedy: The average BLEU score over the test set was 0.009455582363610982

RNN without attention tranining
Epoch 1: loss=5.113278384038115, BLEU=0.0015629711117718618
Epoch 2: loss=4.328808843142695, BLEU=0.0024184759596222022
Epoch 3: loss=4.034494302367019, BLEU=0.00026484280719349797
Epoch 4: loss=3.8479980862497376, BLEU=0.000823970821061077
Epoch 5: loss=3.82716555600642, BLEU=0.0006006164097612194
RNN without attention testing
--beam-width 4: The average BLEU score over the test set was 0.0005166577445610675
--greedy: The average BLEU score over the test set was 0.0021369587943864



Single Head Attention:

LSTM with single head attention training
Epoch 1: loss=4.513680952072966, BLEU=0.0039033364529435836
Epoch 2: loss=3.381194536199686, BLEU=0.005194487113552084
Epoch 3: loss=2.86416574449323, BLEU=0.006252165363894191
Epoch 4: loss=2.4959112524775384, BLEU=0.005658305225775994
Epoch 5: loss=2.2127058205308243, BLEU=0.003620753635717422
LSTM with single head attention testing
--beam-width 4: The average BLEU score over the test set was 0.0067771908201445
--greedy: The average BLEU score over the test set was 0.008013473086308091

GRU with single head attention training
Epoch 1: loss=4.328720054830645, BLEU=0.004815343272555368
Epoch 2: loss=3.259838032037363, BLEU=0.007590225542922517
Epoch 3: loss=2.832847043325771, BLEU=0.007246894337335914
Epoch 4: loss=2.620234337964553, BLEU=0.005980159440337411
Epoch 5: loss=2.407621335985884, BLEU=0.004713360981668308
GRU with single head attention testing
--beam-width 4: The average BLEU score over the test set was 0.006626223265308421
--greedy: The average BLEU score over the test set was 0.00944377799107981

RNN with single head attention training
Epoch 1: loss=5.3208398789140885, BLEU=0.0012111983444763946
Epoch 2: loss=4.7305704904317337, BLEU=0.0012659909076301427
Epoch 3: loss=4.140301101949385, BLEU=0.0013207834707838907
Epoch 4: loss=3.9510798980422925, BLEU=0.002596746701473339
Epoch 5: loss=3.823566768996168, BLEU=0.0015849100786759927
RNN with single head attention testing
--beam-width 4: The average BLEU score over the test set was 0.0018576826471550074
--greedy: The average BLEU score over the test set was 0.002895712885400435



Multihead Attention:

LSTM with multihead attention training
Epoch 1: loss=4.607696820247565, BLEU=0.004225972399757963
Epoch 2: loss=3.46398146778621, BLEU=0.004796426999358815
Epoch 3: loss=2.9643960294323213, BLEU=0.006461851897484704
Epoch 4: loss=2.6308282673386856, BLEU=0.005655117614395254
Epoch 5: loss=2.3855014721496497, BLEU=0.006784102031046591
LSTM with multihead attention testing
--beam-width 4: The average BLEU score over the test set was 0.008275226254145038
--greedy: The average BLEU score over the test set was 0.011297277587159723

GRU with multihead attention training
Epoch 1: loss=4.7811112394815245, BLEU=0.0033338637044727586
Epoch 2: loss=3.7199002722580463, BLEU=0.003405421392111156
Epoch 3: loss=3.3203687751967284, BLEU=0.002396590456350822
Epoch 4: loss=3.0708336248247385, BLEU=0.004625873689087949
Epoch 5: loss=2.9035499370608027, BLEU=0.0033812108878441033
GRU with multihead attention testing
--beam-width 4: The average BLEU score over the test set was 0.0046001852279468495
--greedy: The average BLEU score over the test set was 0.006069525603699923

RNN with multihead attention training
Epoch 1: loss=5.952113920735522, BLEU=0.0027380444675352868
Epoch 2: loss=5.043864832907939, BLEU=0.002399408785413322
Epoch 3: loss=4.848992533350288, BLEU=4.58737397254898e-05
Epoch 4: loss=4.788262949489692, BLEU=0.0003209082955241933
Epoch 5: loss=5.685406360910545, BLEU=0.0
RNN with multihead attention testing
--beam-width 4: The average BLEU score over the test set was 0.0
--greedy: The average BLEU score over the test set was 0.0

In trainings without attention, the best model resulting was GRU, in trainings with singlehead and multihead attention, the best resulting model was LSTM. Though I do not have a specific answer for GRU being the best in training without attention, LSTM being the best in singlehead and multihead attention, is due to the fact that LSTM contains a cell state along with the hidden state and this helps the model learn the connections between words in a better way. One reason for LSTMs being more capable is that they use gates (input, forget, and output gates) that regulate the flow of information.

In my trainings with LSTM, BLEU scores is in an increasing trend as we go from the base model without attention to the model with multihead attention, both in testing with greedy and beam search sampling algorithms. Therefore, I can say that my LSTM model was able to improve with single head attention, and then with multihead attention. 

In my trainings using GRU, the use of singlehead attention does not improve the BLEU score of the GRU model without attention. Further, there is a decrease in the BLEU score results when the GRU model with singlehead attention and multihead attention is compared. 

As for the training with RNN, I can state that the training with singlehead attention improves upon the training without attention, both in greedy and beam search. However, the RNN training with multihead attention does not improve beyond the RNN training with single head attention; which was also the case in the training with GRU. One potential explanation for this can be as follows: I trained all of my 9 model variations for 5 epochs; however, as we increase the number of attention heads in traininig, the number of parameters the model needs to learn is increased, and this might have led the model to not converge properly as in the case of the trainings without attention which require less parameters to be learned. The fact that epoch 5 loss of multihead attention being greater than epoch 5 loss of singlehead attention for both LSTM, GRU and RNN trainings is also supportive for my explanation.

I also think that the BLEU scores of all of my 9 trainings is worse than my expectations. The best BLEU score I was able to get is 0.0112, and it is for training with LSTM-multihead attention configuration. BLEU scores between 0-0.1 is seen as almost nothing is correctly captured in translations. I believe that I did no mistake in the encoder and decoder without attention implementation. While I also think that I did no major mistakes in singlehead attention and multihead attention implementation, I am not 100% confident that my implementation is correct there. However, this still made me think and do research about possible points that may have caused my training results to stay way below my expectations. Here are my arguments:

1) The tokenizer used in this implementation is the word tokenizer, we can see that by inspecting the vocabulary files. Especially considering that we are using Turkish, an agglutinative language, if we used BPE or any other subword tokenizer, I think the results could be better.  
2) I did not tune any of the hyperparameters, such as changing the hidden state size or the learning rate. This might have caused me to end up with a correctly implemented model trained with non-ideal hyperparameters. 

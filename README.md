# AAAI 2024 Workshop W24: [ML4CHM 2024](https://winterlightlabs.github.io/ml4cmh2024/program/index.html)
## ConversationMoC: Encoding Conversational Dynamics using Multiplex Network for Identifying Moment of Change in Mood and Mental Health Classification
This work introduces a unique conversation-level dataset and investigates the impact of conversational context in detecting Moments of Change (MoC) in individual emotions and classifying Mental Health (MH) topics in discourse. In this study, we differentiate between analyzing individual posts and studying entire conversations, using sequential and graph-based models to encode the complex conversation dynamics. Further, we incorporate emotion and sentiment dynamics with social interactions using a graph multiplex model driven by Graph Convolution Networks (GCN). Comparative evaluations consistently highlight the enhanced performance of the multiplex network, especially when combining *reply*, *emotion*, and *sentiment* network layers. This underscores the importance of understanding the intricate interplay between social interactions, emotional expressions, and sentiment patterns in conversations, especially within online mental health discussions.

This work was supported by the Natural Environment Research Council (NE/S015604/1), the Economic and Social Research Council (ES/V011278/1) and the Engineering and Physical Sciences Research Council (EP/V00784X/1). The authors acknowledge the use of the IRIDIS High Performance Computing Facility, and associated support services at the University of Southampton, in the completion of this work.



Loitongbam Singh, Stuart Middleton, Tayyaba Azim, Elena Nichele, Pinyi Lyu, Santiago De Ossorno Garcia. __*ConversationMoC: Encoding Conversational Dynamics using Multiplex Network for Identifying Moment of Change in Mood and Mental Health Classification*__.
<!--
```
@inproceedings{azim-etal-2022-detecting,
    title = "Detecting Moments of Change and Suicidal Risks in Longitudinal User Texts Using Multi-task Learning",
    author = "Azim, Tayyaba  and
      Singh, Loitongbam  and
      Middleton, Stuart",
    booktitle = "Proceedings of the Eighth Workshop on Computational Linguistics and Clinical Psychology",
    month = July,
    year = "2022",
    address = "Seattle, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.clpsych-1.19",
    pages = "213--218",
    abstract = "This work describes the classification system proposed for the Computational Linguistics and Clinical Psychology (CLPsych) Shared Task 2022. We propose the use of multitask learning approach with bidirectional long-short term memory (Bi-LSTM) model for predicting changes in user{'}s mood and their suicidal risk level. The two classification tasks have been solved independently or in an augmented way previously, where the output of one task is leveraged for learning another task, however this work proposes an {`}all-in-one{'} framework that jointly learns the related mental health tasks. The experimental results suggest that the proposed multi-task framework outperforms the remaining single-task frameworks submitted to the challenge and evaluated via timeline based and coverage based performance metrics shared by the organisers. We also assess the potential of using various types of feature embedding schemes that could prove useful in initialising the Bi-LSTM model for better multitask learning in the mental health domain.",
}
  ```
-->

## Proposed Framework
<img src="https://github.com/stuartemiddleton/uos_clpsych/blob/main/image/Pipeline.png" alt="Framework">
<br>

## License

### Data Set: 
The CLPsych data set is proprietary and not shared here. Please contact the competition organisers at clpsych2022-organizers@googlegroups.com to get a copy of its distribution.
### Software: 
 - © Copyright University of Southampton, 2022, Highfield, University Road, Southampton SO17 1BJ.
 - Created By : Tayyaba Azim and Gyanendro Loitongbam
 - Created Date : 2022/05/26
 - Project : SafeSpacesNLP (https://www.tas.ac.uk/safespacesnlp/)

## Installation Requirements Under Ubuntu 20.04LTS 
+ The experiments were run on Dell Precision 5820 Tower Workstation with Nvidia Quadro RTX 6000 24 GB GPU using Nvidia CUDA Toolkit 11.7 and Ubunti 20.04 LTS.
+ Install the following pre-requisite libraries:
```
pip install -U sentence-transformers
pip install gensim
pip install transformers
pip install tensorflow
pip install keras

Package                       Version
----------------------------- --------------------
gensim                        4.0.1
keras                         2.9.0
sentence-transformers         2.2.0
tensorflow                    2.9.1
transformers                  4.20.1

```
## Pretrained Models Required
+ download [fastText embedding vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
```
cd <uos_clpsych_dir>
mkdir dataset
cd <uos_clpsych_dir>/dataset
wget -O wiki-news-300d-1M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip
```


## Sentence Embedding Methods
There are two types of sentence embedding methods considered for this study (Please refer to the paper for detail explaination):
+ *sent_emb*: fastText + SBERT 
+ *sent_score_emb*: fastText + SBERT + Task-specific scores


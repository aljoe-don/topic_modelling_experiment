# lda_pipeline

Experimental Side Project Involving topic modelling and semi-supervised learning.

<br>

## Problem Statement
Though large bodies of text data a readily available, it is difficult to access large datasets of labeled text data for classification tasks. Without labeled data, it is difficult to train models which have a high classification accuracy.

<br>

## Dataset and Inputs
The dataset being considered for this project is the 20_news_groups dataset.
It consists of ~20,000 text documents obtained from an internet forum, each of which belongs to one of twenty different classes. There is approximately equal representation of each class.
The dataset is obtained via download from its host site: http://qwone.com/~jason/20Newsgroups/. It is also available via the Python library sklearn. Note that the version used in this project has automatically separated the train and test sets via a 60/40 split, based on date of forum post (later posts in test set).


The data comprises raw text documents, organized into folders, whereby each folder holds documents of a single class.
A single sample from the dataset is:
```
From: jim.wray@yob.sccsi.com (Jim Wray)
Subject: My Gun is like my Ame
Mark Wilson responding to C.D. Tavares:
MW>|So the laws exist, and the penalties are as you say, but nobody is ever
MW>|prosecuted under these laws. They are "traded away" for easy pleas.
MW>Having such gun laws on the books is still better than nothing.
MW>What would the DA have traded away in order to get the guilty plea if the
MW>gun law had not been in effect.
Our liberty?
Right...don't even think about enforcing the law and imposing the prescribed
penalty....let's hose the citizens instead.
The data contains some artifacts from its internet forum origins, like “reply” markers and From/Subject headers.
```

<br>
<br>

## Solution
The goal of the project is to experiment with methods of reducing the number of labeled training samples required to obtain a high classification accuracy in text classification. This exploration will focus itself around semisupervised learning. Namely, a combination of topic modelling and discriminative classification i.e. SVM, decision trees, etc) will be used to classify text documents, where only a small subset of the training set documents is labelled.

Topic modelling is an unsupervised learning method of finding word combinations within a corpus which represent distinct informational topics within the corpus. A topic is an array of words, each with a weighting which determines its relevance to that topic. As there are 20 classes in the dataset, the topic model will be trained to find 20 topics. Each sample can be assigned a topic distribution, which determines to what extent it belongs in any topic. Topic modelling is effective at determining the relations between documents, (i.e., if two documents are found to belong to the same topic, they contain similar information), however it has the limitation that the topics it finds have no obvious interpretation and won’t necessarily match the pre-defined input classes.

In laymen’s terms, the goal of the project is to explore the following hypothesis. Samples in the same topic belong to the same class. If there are 100 samples in a topic, and we know the labels of, say, 10 of them, we can guess the labels of the rest. In this exploration, the entire training set will be used for training a topic model. Once trained, the topic distribution for each sample will be stored as a 20-dimensional vector embedding. Then a small subset of the data will be sampled, the topic embedding of which will be used to train a supervised learner. In theory, the vector embedding will store information not just on the document, but on the features within the document which most effectively differentiate it from other documents. Thus, it might only require a small number of samples to be able train a classifier to high accuracy.

<br>


# EXPLORING THIS REPO
<hr>

The `notebook` folder contains the raw dataset (~$80Mb$) and teh notebook used to train the LDA model and experiment with its effectiveness.

The `text_processor` folder contains a basic micros-service deployment of the lda model, accessible via REST API. To run the micro-service `cd` into `text_processor` and build the Docker image. Run the Docker container with exposed ports. The endpoint for testing (once Docker is up and running) is `http://127.0.0.1:8080/process`.

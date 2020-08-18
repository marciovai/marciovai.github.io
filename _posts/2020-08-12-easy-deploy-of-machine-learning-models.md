---
layout: post
title: Deploying Machine Learning Models
featured-img: plansee
comments: true
published: false
---

# Introduction

Deploying Machine Learning algorithms into production is rarely a straightforward task. Usually there are multiple variables to be weighted like what type of algorithm, what library was used, how large is the model, how it's predictions are going to be consumed (batch or online inference), what are the business goals, just to name a few. While there is no right or wrong way of doing it some solutions can get quite complex with many steps in the pipeline and different technologies involved, which is one of the reasons for many companies hiring Machine Learning Engineers and Data Scientists, while the first takes care of the deployment and the tech stack and the later of making models. In this post I will present a simple way of deploying models using Docker and Python which works with any library and most platforms, whether you are using AWS, GCP, Azure, bare metal or your local machine.

One of the main advantages of having a simple deploy is that it's a lot easier to debug since there are fewer components involved and it also takes less time to setup and get going. On the other hand the disadvantage is the lack of automation and scalability. But don't get me wrong here, you will still be able to use this deployment methodology for the majority of use cases.

# Architecture

Below is a diagram showing how the architecture will be like once deployed. 

{:refdef: style="text-align: center;"}
![Deploy Architecture](/images/deploy_architecture.jpg)
{: refdef}


Here the API and Model are in the same container to make things simpler but for future improvements it should be relatively straight forward to separate the API from the Model and keep each in it's own container. The API handles all the communication with the outside world while interfacing with the model to get the results from it. Using Docker makes so that deployment can be made in virtually any platform, as long as you are able to run it on the host there shouldn't be any problems. It also works great for making an isolated environment so if it works locally on testing it should behave similarly once deployed somewhere else.

## Model
For this example the model being deployed is a
[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). The problem being solved is a Sentiment Analysis on a balanced sample of 10K tweets I made out of the sentiment140 dataset. The dataset can be found [here](https://github.com/marciovai/Twitter-Sentiment-10K).
The client will send a tweet or a set of tweets inside a JSON on the request body and the server will respond with the corresponding sentiment of each tweet (positive or negative) based on the text.

{:refdef: style="text-align: center;"}
![Twitter Sentiment Analysis](/images/tweet_sentiment_analysis.jpg)
{: refdef}

Before reaching the model, each tweet goes through two steps: 1) Preprocessing following the traditional NLP pipeline which includes: removal of URLs, removal of punctuation, tokenization, removal of stop words, stemming and case-folding.
2) Two features are created based on word frequencies, that is, for each word that appear in the preprocessed tweet, they are summed based on how many times it appeared on both positive and negative tweets.

After processed two features are created, which are the total counts of word frequencies for positive and negative words in the document.

The model outputs a probability between 0 and 1 based on the input features generated for each tweet, which is then filtered to become the forecast labels.

This is a very simple approach towards modeling Sentiment and it usually lags behind other simple approaches like [Naive Bayes](https://web.stanford.edu/~jurafsky/slp3/4.pdf). Also there are other more complete solutions using [Google word2vec](https://code.google.com/archive/p/word2vec/) which are capable of capturing semantic representations and achieve much better performance.

Since the objective of this blog post is to show how to deploy an algorithm rather than develop and improve it, I kept things as simple as possible. If you are working on a similar problem it should be easy to change the Logistic Regression being used for a Neural Network or Vector Spaces and still keep the same project structure for deployment, except of course that a few more steps might be needed depending on the strategy being used.

For referencing, check [this Notebook](https://github.com/marciovai/Twitter-Sentiment-10K/blob/master/Tweet_Sentiment_Analysis_Logistic_Regression.ipynb) to see how the model was developed.

## Setting up the environment
The first step of the deployment is to setup an environment where both the model and the API will run, ideally it should be isolated from the OS of the server so that reproducibility becomes something guaranteed. For the rescue comes **Docker** which does exactly that. Docker is great since it works on the Infrastructure-as-Code paradigm, so the file used to define the environment also becomes documentation and a way to rebuild the environment whenever necessary. Here we will keep things as simple as possible so that building the environment inside the container becomes a solution, not a problem. 
I will leave here a tip for working with Docker: If some particular set of scripts doesn't seen to be working no matter what, you are probably trying to do something in a way that Docker wasn't designed for, try searching for similar solutions to the problem. Things should be very streamlined when using Docker to build an environment.

Below is the Dockerfile that will be used to build the environment.

```Docker
FROM ubuntu:xenial

# update environment packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install software-properties-common -y 

# install pip
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# copy python packages list to container and install
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# create folder where project folder will be attached to during RUN
RUN mkdir external_lib
```

Nothing out of the extraordinary here, just updating the libraries
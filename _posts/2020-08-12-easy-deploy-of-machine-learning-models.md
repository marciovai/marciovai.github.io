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
The first deployment step is to setup an environment where both the model and the API will run, ideally it should be isolated from the server OS so that reproducibility becomes guaranteed. For the rescue comes **Docker** which does exactly that. Docker is great since its build with the Infrastructure-as-Code paradigm in mind, so the file used to define the environment also becomes documentation and a way to rebuild the environment whenever necessary. Here we will keep things as simple as possible so that building the environment inside the container becomes a solution, not a problem. 
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

Nothing out of the extraordinary here, just updating the libraries, installing pip and all the Python libraries will be used from ```requirements.txt```. One thing worth noting here is the ```COPY``` command, which basically takes the file from the local directory where the ```Dockerfile``` being executed is located and copies it to the container.

To build the container call the following command on a terminal where Docker is 
acessible from. This will go through the container definition and build it as ```model_api```.

```bash
docker build -t model_api:latest .
```

## API
For the API, the framework used is [Flask](https://flask.palletsprojects.com/en/1.1.x/api/) which is great for building production ready, but yet simple APIs in Python. Since The one being built here is relatively simple, configuration won't take long plus the source code will be easy to understand. 

The code below instantiates a Flask app on ```api.py```  and adds it to the context of the module where it is located. Notice that we set Debug Mode to ```False``` since we want to use it in a production server.

```Python
# setup flask app
app = flask.Flask(__name__)
app.config["DEBUG"] = False
```

Method below is the main API component - It's the method called when it receives a ```POST``` request from the client, here the route will be ```/api/v1/7Aja2ByCyQ4rMBqA/predict```. 

```Python
@app.route('/api/v1/7Aja2ByCyQ4rMBqA/predict', methods=['POST'])
def predict_tweet():
    # Called when API receives a POST request. It expects a json with
    # a dictionary of tweets, then it calls the model module to generate
    # predictios and returns it

    # Parameters:
    #    json (dict): request dict, should contain ids as keys and tweets
    # as values

    # Returns:
    #    dict (json): where the key is tweet id and value the prediction

    try:
        # get json from request
        json = request.get_json()
        
        # check if request sent an empty payload
        if json != {}:
            # get each id and tweet from request
            ids = []
            tweets = []
            for id, tweet in json.items():
                ids.append(id)
                tweets.append(tweet)

            # call model predict
            result = predict(tweets) 
            
            # generate response json
            if result == []: 
                result = empty_response(ids)
            else:
                result = prepare_response(ids, result)
        else:
            raise Exception('')

        return jsonify(result)
    except Exception as e:
        app.logger.info('Application Error while processing request')
        abort(400)
```

The long string in the middle is used so that the URL becomes dynamic, since there will be no authentication behind the API, this section can be used as a token. For that reason, if you want to deploy this in production I recommend one of the following:

- Implement a proper authentication mechanism using the libraries Flask already has; or
- Deploy it on a server that is inside a private network, like a VPC for example, so that only the IPs you own have access to it; or
- Implement a auto refresh of the API URL token, so that every couple of hours or days it changes.

It's also a good idea to add some extra methods to the API to preprocess the different types of responses and errors that are expected to occur. Feel free to refer to the ones I used [here](https://github.com/marciovai/tweet_sentiment_predictor_api).

## WSGI

Before moving on, there is one more component that needs to be presented which is [gunicorn](https://gunicorn.org/), a powerful production-ready WSGI in Python. The WSGI will be in charge of handling the actual HTTPS connections received from the client to the API and forwarding it the Python code wrote to process the input data. 

The Main benefits of using gunicorn as a WSGI are: 1) It's scalable with support to multithreading out of the box 2) We can use it to process encrypted HTTPS requests without the need of having a proxy like NGINX or Apache.

Without HTTPS encryption, the data contained in the requests to and from the API would be traveling through the web as raw text, making it very easy to be captured by someone ill-intentioned.

Have said that, in order to use HTTPS we need to have a _Certificate_ so that we can encrypt the data we send, and only the receiver can decrypt it. Assuming a UNIX-based terminal and OpenSSL is installed, the command below will create a ```server.key``` and ```server.crt``` after following the on-screen instructions.

```bash
openssl req -newkey rsa:2048 -nodes -keyout server.key -x509 -days 365 -out server.crt
```

Since this is only an example it's perfectly fine to use a self signed certificate (or if it's running inside a private network). Otherwise, I highly encourage getting one from a trusted authority.

With the certificate and keyfile to encrypt requests at hand, there is one more step that needs to be done to configure gunicorn so that it can be used by Flask.

The way process will be chained to run the entire API is the following: 1) gunicorn starts as a system-wide process 2) gunicorn starts the Flask app to run on the processes encapsulated by it.

For this to work, we need to pass the API context to gunicorn once it starts. The code snippet below defines the app context on ```wsgi.py``` that the WSGI needs to work.

```Python
from api import app

# this file is used to pass the app context to WSGI Gunicorn

if __name__ == "__main__":
    context = ('server.crt', 'server.key')
    app.run(host='0.0.0.0', debug=False, ssl_context=context, threaded=True)
```

The Flask app defined before will be imported and executed with the passed parameters. Once gunicorn is called on the command-line, it will chain this Flask process inside it's threads to encapsulate the Flask app.


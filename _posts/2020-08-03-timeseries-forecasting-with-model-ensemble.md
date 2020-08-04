---
layout: post
title: Timeseries Forecasting with Model Ensemble
featured-img: plansee
comments: true
---

# Introduction

When it comes to performing Timeseries forecasting and analysis things can get challenging depending on the nature of the event being modeled, the data and the business requirements for the task at hand. For example a retail chain might want to know future sales numbers to better prepare stock/planning. A university studying weather events needs to know what to expect from specific atmosphere indicators to predict the possbility of experiencing a storm. A financial institution needs to be up to date with the market what could be observed in the days to come.

In the past I've done multiple Timeseries Forecasting projects in different companies and in this post I will share one possible solution that I came across. Which brings us to my next point, Timeseries modeling can be hard to tackle depending on the data available and what results are desired. For example, it's always possible that a given event is mostly random by nature and no set of features can help model it with decent accuracy or maybe the forecast horizon for a particular problem is too long to achieve good enough results to be used in practice with any algorithm and available data. 

Before we begin I would like to highlight the importance of starting small and simple before going big and complex. It can be very frustrating to spend a lot of time trying to fit a fancy LSTM architecture to solve a Timeseries problem to only find out later that an ARIMA based algorithm performs as good or maybe even batter, so take this tip from someone that did spend a lot of time trying different types of solutions. A great place to start is the [Facebook Prophet](https://facebook.github.io/prophet/), an incredibly powerful library for Timeseries analysis and modeling with pre-built ARIMA and Timeseries decomposition models.

For our solution today the algorithm of choice is the GBM (_Gradient Boosting Machine_), more specifically the LightGBM implementation of it. You've probably stumbled across it already since it's very popular on Kaggle competitions, which serves as testimony of its success for Timeseries modeling. 
One of the strenghts of tree-based algorithms when used for Timeseries is its ability to use categorical data. For example imagine you would try to forecast sales from multiple stores across the country. The tree architecture will be able to use this information to build the trees/predictors and create this sense of "hierarchy" within a single model. As it turns out this is a category of problems in itself, which is called Hierarchical Timeseries, I will leave here [Optimal combination forecasts for hierarchical time series](http://webdoc.sub.gwdg.de/ebook/serien/e/monash_univ/wp9-07.pdf) as an interesting read if you want to learn more about the topic.

Also, before we being: Don't take this as the state-of-the-art solution for any Timeseries problem because there is no such thing, at least for now. Instead, use this as an additional tool or modeling approach to add to your arsenal for solving Timeseries tasks. Now let's get to the modeling!

# The Model(s)

As mentioned before the algorithm we will be using is the GBM, basically the idea is to create an ensemble of multiple trees, with each tree forecasting at _t+i (i=1,..., N_) where _N_ is the last date in our forecast horizon. So with that if for example we have data sampled per week and want to forecast 12 weeks ahead, a total of 12 models will be needed where each subsequent model forecasts one week ahead of the previous one. 

The image below should help illustrate what we want to achieve in terms of model architecture and forecast procedure.

{:refdef: style="text-align: center;"}
![Model Ensemble](/images/gbm_ensemble.jpg)
{: refdef}

So basically we will train N models, where N is how many time steps are needed to achieve the desired forecast horizon. The main advantage of this approach is that each model becomes more flexible to learn the different time steps than it would if we were using only a single model. This characteristic is specially useful when dealing with multivariate timeseries or other more complex events.

The GBM is essentially an ensemble in itself, since it relies on *boosting* to build a sequence of trees based on the input features and it tries to improve accuracy after each subsquent tree by splitting it further. Here in our approach, we are basically complementing the boosting strategy by adding also *bagging* which is another type of model ensembling. If you are curious, [scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html) has a good read on the topic of bagging & boosting ensembles.

## Training Approach

The logic behind our modeling approach is fairly simple, now let's get to the more practical aspect and see how this is actually achieved in practice. 

As some may have guessed, all that we have to do is shift the labels in the training data at each new model created before training it. With that we are basically forcing the model to learn how to predict at N steps ahead. So model _N=5_ predicts the labels for 5 weeks in the future from the observation. 

I was surprised when I first implemented this and saw how good it performed, not because I didn't trust the model capability but more for how simple the logic behind it really is. Things like this reminds us of the function approximator nature of Supervised Machine Learning models, and that as long as there is enough data available many approaches can be used to solve a given problem. 

The image below shows an example of the data after shifting the labels one timestep iteractively. For this approach this is performed N times, where N is how many timesteps we need to perform in order get the entire forecast horizon.

{:refdef: style="text-align: center;"}
![Data Labels at t+n](/images/data_t_n.jpg)
{: refdef}

One thing to be mindful here is that for each timestep taken a GBM model instance will be created, trained and stored (in memory or disk). Depending on the size of the dataset and the forecast horizon, it might be necessary to do this in batches. A benefit of this implementation is that it should be fairly easy to parallel process the entire procedure across the N dimmension since at each iteration the model being created does not depend on the previous, the only thing that needs to be distributed across all processes/threads is the input data feed.

# Implementation

## The dataset

The dataset we will be using is the [Sunspot](http://www.sidc.be/silso/datafiles) dataset.

> Sunspots are temporary phenomena on the Sun's photosphere that appear as spots darker than the surrounding areas. They are regions of reduced surface temperature caused by concentrations of magnetic field flux that inhibit convection
>
> -- <cite>[Wikipedia](https://en.wikipedia.org/wiki/Sunspot)</cite>

It goes from 1749-01-01 to 1983-12-01 aggregated in months, giving us a total of 2820 observations, which would be a pretty decent sized dataset but remember that its about an astronomical event. Regardless, its more than enough for us to explore, build our model ensemble and measure the accuracy later on.

{:refdef: style="text-align: center;"}
![Sunspot Dataset](/images/sunspot_dataset.JPG)
{: refdef}

As you can see the data is somewhat noisy and with high variance, it also has varying seasonal effects. This proves for an interesting test for our modeling approach, since the model needs to learn to capture the different seasonal effects present in the data in order to give a proper forecast.

Below is a de-noised version of the same plot using a Moving Average with a window of 12, transforming the data more or less in an early plot. 

{:refdef: style="text-align: center;"}
![Sunspot with Rolling Average](/images/sunspot_dataset_average.JPG)
{: refdef}

## Data preprocessing

First let's get the dataset and load it to memory. If you are using a Notebook like myself you can download the data on the current directory by executing the command below inside a cell, otherwise you can just follow the link and save it on your computer manually.
```bash
# download dataset Monthly Sunspots
!wget -O monthly-sunspots.csv https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv --quiet
```

Now let's load it with Pandas.
```python
# load data
data = pd.read_csv("./monthly-sunspots.csv")
```

Back to the data preprocessing - let's transform the dataset so that the algorithm can extract as much information as it can to lower the training error. 
When dealing with Timeseries modeling, adding [lags](https://math.stackexchange.com/questions/2548314/what-is-lag-in-a-time-series) as features is always a no-brainer. Except if the model being used is an LSTM which is capable of learning the time-dependency in the data without being explicitly so, our tree-based model will benefit a lot from having this extra set 
of features added to the input, so we will proceed and create them.

```python
# set lags to be created as extra features for seasonality
lags = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 40, 48]

# create lag features
for lag in lags:
  data['lag_' + str(lag)] = data.Sunspots.shift(lag)
```

After testing, I found that the lags created above are a good combination for our task at hand. It's always good to test different lags with varying timesteps between them since these usually are the most useful features for the algorithms to capture the series autocorrelation.

Let's also create some extra features related to the seasonality that the timeseries have and the cycles present in the data. 

```python
# add seasonality features
# based on date
data['month'] = data.date.dt.month
data['year'] = data.date.dt.year
data['quarter'] = data.date.dt.quarter

# based on series sequence
data['seq'] = list(data.index + 1)
```

Everything should be fairly straightforward here, the only notable mention is the ```seq``` feature, which consists on only the number of each entry in the dataset, this could as well be t he pandas.DataFrame index and is also added here in the hopes helping the model.

Before moving further it's a good idea to create a full copy of our DataFrame for safe keeping and also filter out the rows we won't be needing for now (on the original DataFrame).

```python
 # deep copy full dataset
data_cp = data.copy(deep=True)

# set training data and first entry of test data
data = data[data['date'] <= pd.to_datetime('1981-01-01')]
```

Next let's talk about our *forecast horizon*, here I chose to forecast for 24 months in the future and while it's true that this is just a small fraction of the dataset, it should be enough to show the model performance and capabilities. 

The dates in the forecast horizon are from 1981-01-01 to 1982-12-01, which amounts to 24 months.

Now we will perform a very important preprocessing step in the dataset - de-trending the Timeseries. If you look at the second plot with the Moving Average it's possible to notice a trend overall in the dataset. Unfortunately the GBM wasn't capable to model the trend after a few attemps, so I decided to remove it manually. So the steps are the following regarding the trend:

1) Remove trend from all data
2) Train model
3) Generate predictions
4) Add back trend to data

It's for that reason that we saved a copy of the data in the previous step, without it will not be possible to add back the original trend to the forecast.

```python
# de-trend timeseries by differentiating (can also be achieved by using diff() from Pandas)
data['Sunspots'] = data['Sunspots'] - data['Sunspots'].shift(1)
```

Now the entire dataset is stationary and should be a lot easier for our model! Here I would like to make a honorable mention to [Facebook Prophet](https://facebook.github.io/prophet/) once again, it already has a built-in model to decompose the Timeseries and get the different trends present in data, and it's also very easy to plot so I definetely recommend giving it when working with Timeseries.

Time to create the train and test sets. here you will notice that the ```test``` contains only a single row, and this is all that we will be using during inference time. Since we are training an ensemble of algorithms capable of going from _t+i_ all the way to _N_, the algorithms need only the starting point to be able to deliver the forecast of it's specific time step. So there is no need to feed back the forecast of the previous model to the next one in this case.

```python
# set train and test sets, here test set is just a single row (1981-01-01)
# we will start from this date and forecast 24 months in the future
train = data[data['date'] < pd.to_datetime('1981-01-01')]
test = data[(data['date'] >= pd.to_datetime('1981-01-01'))]
```

Since we created lags as features, there is one more thing we need to do: remove the NaN entries from the first data points. These occur because for example at the first entry in the dataset there is no way of knowing the previous lags and filling with zeros could potentially cause us problems.

```python
# drop NaN, which are the first 2 years of data (due to creation of lags)
train.dropna(axis=0, inplace=True)
```

Good to go! Next let's define our training loop to create our model ensemble!

```python
# training loop to do final data processing and train the models

# set GBM params
# aside from the ones task-specific, everything will be left out as default
param = {'objective': 'regression', 
         'metric': 'mse'}

# create dictionary to store all the models for each month 
models_dict = {}

# iterate from 1 to 24, which will create 24 GBMs, for the 24 months of 
# forecast horizon
for i in range(1, 25):
  print("Fitting model {}".format(i))

  # shift label on training set
  # by doing this we will force the model to learn how to forecast at t+i
  train_iter = train.copy()
  train_iter['Sunspots'] = train.Sunspots.shift(-i)

  # drop last i rows from training sets
  # these have NaN value due to the shift made to the labels
  train_iter.dropna(axis=0, inplace=True)

  # define LightGBM Dataset class
  # create training set
  train_data = train_iter.loc[:, ~train_iter.columns.isin(['date', 'Sunspots'])]
  train_data = lgb.Dataset(train_data, label=train_iter.Sunspots, feature_name=list(train_data), categorical_feature=['month', 'year', 'quarter'], free_raw_data=False)

  # model training
  # set number of training loops
  num_round = 100
  model = lgb.train(param, train_set=train_data, num_boost_round=num_round)

  # save model to dict
  models_dict.update({i: model})
```

So in the code above we are basically iterating over each month, shifting the dataset labels accordingly, removing the NaN values, fitting the model and adding it to a dictionary of models.

Now we have ```models_dict```, which have each all our models in it, next we just need to generate the predictions using them.

```python
# Forecast for upcoming 24 months using test data (entry for 1981-01-01)
# all 24 models created above will be used, each one for t+i
forecast = []
for i in range(1, 25):
  # get entry from test set, which is just a single row and predict
  test_data = test.iloc[0, ~test.columns.isin(['date', 'Sunspots'])]
  # get model instance from current iteration and reshape input data
  prediction = models_dict[i].predict(test_data.values.reshape(1, -1))
  forecast.append(prediction[0])
```

Almost finished, last step is to add back the trend to the data (including the predictions) and check out the results.

```python
# set date range for filtering test dataframe
start_forecast = '1981-01-01'
end_forecast = pd.to_datetime(start_forecast) + relativedelta(months=+24)

# create test_df from data_cp which has all the original data in it (no differentiation)
test_df = data_cp[(data_cp['date'] >= start_forecast) & (data_cp['date'] < end_forecast)]

# add forecast to DataFrame before adding back trend
test_df['Sunspots'] = forecast

# create single dataframe with all data (train + test)
data = data.append(test_df)

# undo differentiation on data and save results on data DF
x, x_diff = data_cp['Sunspots'].iloc[0], data['Sunspots'].iloc[1:]
data['Sunspots_trended'] = np.r_[x, x_diff].cumsum().astype(int)
```

Now that the trend is back, let's add the date labels to the data and plot!
```python
# filter out all 24 months from forecast and get the respective DataFrame
data_test_dates = data[data['date'].isin(test_df.date)]

# skip first row since it's duplicated because of the preprocessing done
data_test_dates = data_test_dates.iloc[1:]

# add original labels from data_cp, which was untouched from the start
data_test_dates['label'] = data_cp[(data_cp['date'] >= start_forecast) & (data_cp['date'] < end_forecast)]['Sunspots'].values
```

Comparing our forecast (```Sunspots_trended```) and the actual labels

{:refdef: style="text-align: center;"}
![Results 1](/images/results_1.JPG)
{: refdef}

{:refdef: style="text-align: center;"}
![Results 2](/images/results_2.JPG)
{: refdef}

As we can see the model did a pretty good job overall in modeling the series, including the seasonality observed near the end of the last months.

Lastly, let's take a look at the feature importance from the models and average them out to see which ones had the most influence while forecasting. The code below will get the feature importance from each of the 24 models and save them on a DataFrame for plotting

```python
# get feature importance from all 24 models
feature_importance_df = pd.DataFrame()

for index, model in models_dict.items():  
  iter = pd.DataFrame(data=model.feature_importance()).T
  iter.columns = model.feature_name()
  iter.index = [index]
  feature_importance_df = feature_importance_df.append(iter)
```

{:refdef: style="text-align: center;"}
![Results 2](/images/results_2.JPG)
{: refdef}
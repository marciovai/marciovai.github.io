---
layout: post
title: Building a Recommendation System in Python
featured-img: banff
comments: true
---

# Introduction
With the rise of companies that rely heavily in their ability of recommending content to their Users, the interest for tools and techniques that can be used for making better content matching also grew. In this post we will walk through the steps towards building a simple Recommendation System in Python that works based on past user-item interactions and can be used in a range of different applications.  
In this post we will see to build a recommendation system in Python using Cosine Similarity for a _Collaborative-Filtering_ approach.

Don't worry about copying and pasting the code from this article, I will leave a Python file as well as a iPython Notebook with the full source code available at this [Github Repo](https://github.com/marciovai/recommendation_system).

# How it works
There are two main paths that we can choose in order to solve our problem. They are the *Content-based* and *Collaborative-Filtering*. Both have it's strenghts and drawbacks as well as varying levels of complexity. Let's take a quick look on each of them to familiarise ourselves with the topic.

## Content-based
Content-based recommendation is usually the most accurate one when both Users and Items are well known with descriptive data as well as previously recorded interactions between both. The drawback is that the implementation for this approach is relatively more complex since most of the times it requires combining different types of data efficiently.

What we want to do here is identify which Items are similar to each other based on what we know and recommend them to Users that liked similar Items and also we can recommend Items from Users that have a similar profile, so there is a lot to gain here in terms of data.

{:refdef: style="text-align: center;"}
![Collaborative Filtering](/images/collaborative-filtering-example.jpg)
{: refdef}

For example suppose we want to apply our algorithm to an e-commerce store, data available for our problem could be:
- Item: text description, attributes, image
- User: profile data, navigation behavior
- User x Item: pageviews, wishlist, add to cart, purchases

In order to sucessfully build a recomendation algorithm based on content the data listed above needs to be combined effectively. Usually the best approach is rellying on some sort of model ensemble with Neural Networks and Embedding, altough unfortunatelly often the network becomes complex to converge.


## Collaborative Filtering
For the Collaborative Filtering approach we free ourselves from having to combine multiple data sources and types together in a simple algorithm. Here we rely purely on interaction data, which is the same User x Item we saw for the Content-based approach.

In practice what happens is that we measure how similar Users are from each other based on the Items they interact with. For example actions like pageviews, add to wishlist, cart or purchase are considered as the User expressing interest on particular items.

{:refdef: style="text-align: center;"}
![Collaborative Filtering Example](/images/collaborative-filtering-example2.jpg)
{: refdef}

The simplest way to implement this similarity measure is to simply use **Cosine Similarity** across the rows of a User x Item matrix. Let's elaborate a bit on that last sentence and understand what calculations we are doing and what are the expected results.

#### Cosine Similarity
Measures the similarity of two vectors, this is done by calculating the cosine angle between them. The benefit of using this method is that it's invariant to vector size, so for example imagine we have more entrances in one vector than in the other which would make one vector larger than the other. This difference in magnitude doesn't affect Cosine Similarity since it only considers the angle between the two vectors instead of their sizes as a measure of similarity.

Below is the formula we will be using, for the sake of reference:

{:refdef: style="text-align: center;"}
![Cosine Similarity Formula](/images/cos_sim.png)
{: refdef}

Even though we won't be needing to implement it ourselves since Python libraries already have it, it's always good practice to understand what we are doing. 

As you can see in the formula, we normalize by the module of the product of each vector. This helps ensure that our calculation remains invariant to vector size.

Other benefit is that values are already normalized between -1 and 1 for opposite orientation and exact same orientation respectively.

{:refdef: style="text-align: center;"}
![Cosine Similarity Chart](/images/cos_similarity_chart.jpg)
{: refdef}

## Dataset
As mentioned previously, Collaborative-Filtering relies on interaction data so we need a set of recorded interactions between Users and Items. Luckily for us there is a dataset that does exactly that - the MovieLens 100K dataset composed of 100.000 individual movie ratings ranging from 1 to 5, made by 943 users in 1682 movies.
This is probably the best publicly available dataset for working on recommendation systems, it's based on real data and is already preprocessed for us.

It's important to note that even though the dataset we will be using is comprised of user ratings on movies, our Recommender System will still be easily adaptable to other types of recommendation problems. At the core of what we are doing, we are simply using Cosine Similarity as a way to determine similarity between two different vectors (users) based on their features (item interactions).

Without further ado let's get to it!

## Development

### Preparing the data
Since our goal is to use **Cosine Similarity** to measure how close Users are from each other, we need to transform our dataset from a dense to a sparse representation. In order to achieve that each User needs to be represented by a single row in the dataset so that the columns are the ratings given by the Users to each different movie.

But first, let's get our MovieLens 100K Dataset!  
The code below will download the dataset from the public repository, extract it and parse some of the columns to the correct datatype.

```python
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import zipfile

# Download MovieLens data.
print("Downloading movielens data...")

urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
zip_ref = zipfile.ZipFile('movielens.zip', "r")
zip_ref.extractall()
print("Done. Dataset contains:")
print(zip_ref.read('ml-100k/u.info'))

# Load each data set (users, movies, and ratings).
ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

# The movies file contains a binary feature for each genre.
genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movies_cols = [
    'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
] + genre_cols

movies = pd.read_csv(
    'ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

# Since the ids start at 1, we shift them to start at 0.
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x-1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x-1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x-1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

# Get date from unix timestamp
ratings['date'] = pd.to_datetime(ratings['unix_timestamp'], origin='unix',unit='s').dt.date

# remove unix date since we already have it in  a human-friendly format
ratings.drop('unix_timestamp', axis=1, inplace=True)
```

Let's see how our data looks by running:
```python
ratings.head()
```
{:refdef: style="text-align: center;"}
![Movie Lens 100K](/images/movie_lens_100k.jpg)
{: refdef}

So we have ```user_id```, ```movie_id``` and ```rating``` which is what we needed to solve our problem plus the newly added ```date``` which we will be using to separate the training and test sets later on.

Before moving on to the algorithm itself there is one more step to take care of, and it's the most important one: transforming the dataset from *dense* to *sparse*. We can use pandas **pivot** to solve this.

Cool, now we can transform our data into a pivot table with the following command:
```python
data_pivot = ratings[['user_id', 'movie_id', 'rating']].pivot(index='user_id', columns='movie_id').fillna(0).astype(int)
```

Next step we will transform our sparse DataFrame into a [SciPy row-sparse matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).

```python
from scipy import sparse

# create scipy sparse from pivot tables
data_sparse = sparse.csr_matrix(data_pivot)
```

You may be asking whats the point of doing that transformation since we can already use our currently NumPy ```ndarray``` as is. 

This brings us to one of the drawbacks of this method - We need to have the entire dataset in memory to be able to compute distances among all Users, even though we could implement a process to load Users in batches instead of all at once because we need to do NxN computations in order to calculate all distances (where N = # of Users) there will still be a big IO bottleneck.

In our case we should be able to fit the entire dataset in memory just fine since we only have 943 Users and 1682 Movies total but if the User base was composed of hundreds of thousands of Users and tens of thousands of Items (which it usually is in practice) things would've been different.

Our SciPy implementation is very memory efficient for storing sparse datasets like the one we have, which has many zero entries per row.

Now we are good to go, our Dataset is ready so lets calculate the distances!

### Calculating the distance among Users

We could write our own implementation to calculate the Cosine Similarity for each row in the dataset, but luckly for us *sklearan* already has it implemented so we will go with it.

```python
from sklearn.metrics.pairwise import cosine_similarity

# calculate similarity between each row (user x movies)
similarities_sparse = cosine_similarity(data_sparse, dense_output=False)
```

Here we will use ```dense_output=False``` to have the output as a SciPy sparse matrix, this is a step that we are taking to make sure that our matrix fits in memory, otherwise the output would be a numpy ```ndarray``` which isn't as efficient for storing large sparse datasets.

The shape of our  ```similarities_sparse``` is ```(# of users, # of users)``` and the values are the similarity scores computed for each User against every other User in the dataset.

Next for every User we need to get the *top K* most similar Users so that we can look at which Movies they liked and make suggestions - that's where the actual **Collaborative Filtering** happens.

The method ```top_n_idx_sparse``` below takes as input a ```scipy.csr_matrix``` and returns the *top K* highest indexes in each row, thats where we get the most similar Users for each User in our Dataset.

```python
# returns index (column position) of top n similarities in each row
def top_n_idx_sparse(matrix, n):
    '''Return index of top n values in each row of a sparse matrix'''
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx

user_user_similar = top_n_idx_sparse(similarities_sparse, 5)
```

Here I decided to pick the top 5 most similar Users for each User since it should be enough for getting recommendations, but feel free to increase the value for K if your particular problem requires it.

### Process data for easier visualization
Now we have our set of recommendations for each user stored in a sparse matrix, next we want to transform the data to a format easier to preprocess.

```python
# transforms result from sparse matrix into a dict as {matrix_index: [user_1, user_2,..., user_5]}
user_user_similar_dict = {}
for idx, val in enumerate(user_user_similar):
        user_user_similar_dict.update({idx: val.tolist()})
```

Our dict has as keys the row numbers from the sparse matrix and user lists as values, so what we need to do next is transform the key indices into the respective user ids from our dataset, thus matching the recommendations with the right user ids.

```python
# gets actual user ids from data based on sparse matrix position index
similar_users_final = {}
for user, similar_users in user_user_similar_dict.items():
    idx = data_pivot.index[user]
    values = []
    for value in similar_users:
        values.append(data_pivot.index[value])

    similar_users_final.update({idx: values})
```

So we fixed the keys of our dict to be the actual user ids from the dataset instead of the index from the sparse matrix. Last step is to obtain the actual movie ids from our list of users.
For each user in the dataset we have a list of 5 most similar users, we will use these to get our recommendations.

The code below is going to get a sample from the Movies that the similar users interacted with and store them into a dictionary. Essentially what we are doing here is transforming our User x User dict into a User x Movie one.

There are a other few things happening in the code below that are worth noting:
- We get only a sample of movies from each similar user (10 in this case) because this dataset is somewhat dense in the sense that on average users rated at least 20 movies each. By taking samples we get a balanced set of movies from all top k the similar users.

- Some extra logic is needed since our dataset is about movie ratings (from 1 to 5 starts), we want to make sure that we recommend to users movies that other similar users liked, so we filter by ```rating >= 3```. Unfortunately there could be the case where users only rated movies lower than that, so we add some extra code to make sure that every user gets at least a single recommendation.

```python
# transforms list of users: [similar_user1, similar_user2] into list of user: [job1, job2]
user_movies = {}
for user, similar_users in similar_users_final.items():
    # remove the user itself from similar_users (since cos_sim(user_1, user_1) is 1)
    try:
        del similar_users[similar_users.index(user)]
    except:
        pass
    # get movie ids from list of movies rated by similar users.
    # also apply extra logic to get the most high rated movies from the similar users
    movies_rec = ratings[(ratings['user_id'].isin(similar_users)) & ratings['rating']>=3]
    if movies_rec.empty:
      movies_rec = ratings[(ratings['user_id'].isin(similar_users)) & ratings['rating']>=2]
    if movies_rec.empty:
      movies_rec = ratings[(ratings['user_id'].isin(similar_users)) & ratings['rating']>=1]
    movies_sample = movies_rec.sample(n=10, random_state=33)['movie_id'].values
    user_movies.update({user: list(set(movies_sample))})
```

Awesome, now we are close to the end result we were looking for, last step is to store this data into a DataFrame for easier visualization and manipulation.

### Looking at the results

```python
# transform dictionary into list of tuples and save on DataFrame
user_movie_tuple = [(user, movie) for user, user_movies in user_movies.items() for movie in user_movies]
user_movie_df = pd.DataFrame(user_movie_tuple, columns=['user_id', 'movie_id'])
```

Next we want to add the movie names to the DataFrame with the recommendations by joining it with the ```movies``` one:
```python
rec_batch = pd.merge(user_movie_df, movies[['movie_id', 'title']], on='movie_id', how='left')
```
Looking at the recommendations made for a specific Users:

```python
rec_batch[rec_batch['user_id']=='99'][['movie_id', 'title']]

movie_id	title
288	        Evita (1996)
257	        Contact (1997)
312	        Titanic (1997)
321	        Murder at 1600 (1997)
690	        Dark City (1998)
311	        Midnight in the Garden of Good and Evil (1997)
299	        Air Force One (1997)
320	        Mother (1996)
325	        G.I. Jane (1997)
```

Looks like the user_id 99 likes Drama movies! With this DataFrame at hand we can know recommend Movies to all the Users in the Dataset based on their behavior. Once a new User joins we won't be able to make recommendations right away ([Cold Start](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems))), but once the User starts rating Movies previously watched we will be able to measure it's similarity with other existing Users and make the recommendations based on that.

The full source code as well as a iPython Notebook can be found on this GitHub [here](https://github.com/marciovai/recommendation_system).

Thanks for reading and if you have any trouble implementing this solution or have any feedback in general feel free to leave a comment below or contact me!
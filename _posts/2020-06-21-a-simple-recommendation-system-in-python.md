---
layout: post
title: A simple Recommendation System in Python
featured-img: sleek
comments: true
---

# Introduction
With the rise of companies that rely heavily in their ability of recommending content to their Users, the interest for tools and techniques that can be used for making better content matching also grew. In this post we will walk through the steps towards building a simple Recommendation System in Python that works based on past user-item interactions and can be used in a range of different applications.

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

In practice what happens is that we measure how similar Users are from each other based on the Items they interact with. Actions like pageviews, add to wishlist, cart or purchase are considered as the User expressing interest on them.

{:refdef: style="text-align: center;"}
![Collaborative Filtering](/images/collaborative-filtering-example2.jpg)
{: refdef}
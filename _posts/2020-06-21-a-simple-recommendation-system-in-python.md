---
layout: post
title: A simple Recommendation System in Python
featured-img: sleek
comments: true
---

# Introduction
With the rise of companies that rely heavily in their ability of recommending content to their Users, the interest for tools and techniques that can be used for making better matching content to platform users also grew. In this post we will walk through the steps towards building a simple Recommendation System in Python that works based on past user-item interations and can be used in a range of different applications.

# How it works
There are two main paths found in the litereture that we can go to in order to solve our problem. They are the *Content-based* and *Collaborative-Filtering*. Both have their strenghts and drawbacks with different levels of complexity.

## Content-based
Content-based recommendation is usually the most accurate one when both Users and Items are well known and there is a good amount of descriptive data available for both, as well as previously recorded interactions. The drawback is that the implementation is relatively more complex since most of the times it requires combining different types of data efficiently.
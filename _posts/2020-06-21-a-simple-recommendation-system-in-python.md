---
layout: post
title: A simple Recommendation System in Python
featured-img: sleek
comments: true
---

# Introduction
With the rise of companies that rely heavily in their ability of recommending content to their Users, the interest for tools and techniques that can be used for making better content matching also grew. In this post we will walk through the steps towards building a simple Recommendation System in Python that works based on past user-item interactions and can be used in a range of different applications.

# How it works
There are two main paths found in the litereture that we can go to in order to solve our problem. They are the *Content-based* and *Collaborative-Filtering*. Both have their strenghts and drawbacks with different levels of complexity.

## Content-based
Content-based recommendation is usually the most accurate one when both Users and Items are well known with descriptive data as well as previously recorded interactions between both. The drawback is that the implementation for this approach is relatively more complex since most of the times it requires combining different types of data efficiently.  
{:refdef: style="text-align: center;"}
![Collaborative Filtering](/images/collaborative-filtering-example.jpg)
{: refdef}
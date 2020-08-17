---
layout: post
title: Deploying Machine Learning Models
featured-img: plansee
comments: true
published: false
---

# Introduction

Deploying Machine Learning algorithms into production is rarely a straightforward task. Usually there are multiple variables to be weighted like what is the type of algorithm, what library was used, how large is the model, how it's predictions are going to be consumed in production (batch or online inference), what are the business goals, just to name a few. While there is no right or wrong way of doing this some solutions can get quite complex with many steps in the pipeline and different technologies involved, which is one of the reasons for many companies hiring Machine Learning Engineers and Data Scientists, while the first takes care of the deployment and the tech stack and the later of making models. In this post I will present a simple way of deploying models using Docker and Python which works with any library and most platforms, whether you are using AWS, GCP, Azure, bare metal or your local machine.

One of the main advantages of having a simple deploy is that it's a lot easier to debug since there are few components involved and it also takes less time to setup and get going. On the other hand the disadvantage is the lack of automation and scalability. But don't get me wrong here, you will still be able to use this deployment methodology for 90% of the use cases.

# Architecture

Below is a diagram showing how the architecture will be like once it's deployed. 

{:refdef: style="text-align: center;"}
![Deploy Architecture](/images/deploy_architecture.jpg)
{: refdef}


Here we will keep the API and the Model in the same container to make things simpler but for future expansion it should be relatively straight forward to separate the API from the Model and keep each in it's own container. The API handles all the communication with the outside world while interfacing with the model to get the results from it. Using Docker makes so that deployment can be made in virtually any platform, as long as you are able to run it on the host there shouldn't really be any problems. It also works great for making an isolated environment so if it works locally on testing it should behave similarly once deployed in production.

## The Model
Here I will choose to deploy a LightGBM model
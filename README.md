# BoomBikes_CaseStudy
> Assignment to apply knowledge of Linear regression.


## Table of Contents
* [General Info](#general-information)
* [Built With](#built-with)
* [Getting Started](#getting-started)
* [Steps Followed during analysis](#steps-followed-during-analysis)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.

A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state.

In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.

They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

Which variables are significant in predicting the demand for shared bikes.
How well those variables describe the bike demands
Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors.

## Built with
Pyhton

## Getting Started
The data given contains the information of 2018 and 2019 data of number of bikes hired, counts provided for different months, different seasons, weather situations and temperature mainly.

## Steps Followed during analysis
* Reading and understanding the data

* Preparing the data for modelling

* Training the data

* Residual analysis

* Prediction and applying model on test data

## Technologies Used
- Language - Python
- Library - Numpy
- Library - Pandas
- Library - matplotlib
- Library - seaborn
- Library - math
- Library - statsmodel
- Library - sklearn

## Conclusions
- R-squared is 83.6% in train set and 81.8% on test set
- Adjusted R-squared is 83.2% in train set and 80.9% in test set
- These values are acceptable and hence we can say that the model is best!
- These are the key variables that boombike need to give more focus
As te company grows we can expect more usage & more revenue in future.
Temperature has direct and strong impact on count of BoomBikes users.
Windspeed impacts the weather, higher te windspeed, bad is the weather, negative corelation with the user count.
Winter seems good, in rainy weather count decreases
When cloudy and misty count decreases

## Acknowledgements
- Tutorial video on EDA, Linear regression, simple linear regression and multiple linear regression
- Weekend live session on the assignment details, linear regression basics and doubt resolution
- This project was based on (https://learn.upgrad.com/course/3611/segment/21307/168560/516120/2650839).


## Contact
Created by [@Rahulkishoreraj] - feel free to contact me!

# A Machine Learning Model using TensorFlow to Describe the Market Price of Cryptocurrencies: The Case of Bitcoin.
### Authors
* **Author/Researcher:** David P Lopez
* **Faculty Advisor & Author:** Professor Eric Hernandez PhD *Candidate*
* **Miami Dade College STEM Research Institute**

## Abstract
### Background
Digital currencies have led to an evolution in how society thinks about the abstract concept of money. This paradigm shift in the accounting of transactions, and what it means to make a transaction with a secure store of value, as a secure and private medium of exchange, that is also engineered with a distributed and immutable general ledger, and protected by RSA encryption in a digital economy riddled with persistent threats against the frictionless flow of digital payments, is what we as the authors of this research, are attempting to quantify and discuss in order to better understand why and how a digital currency like Bitcoin, obtains the inherent value and prices that our society has observed in its short history. 

### Objectives
The objective of this body of work is to leverage the Python programming language and Open Source Machine Learning Libraries like Google's TensorFlow, to find features within a specific data set related to Bitcoin and other correlating asset classes, that we can observe and use to build a model using linear regression techniques to find a set of features that have a high degree of impact on the price of Bitcoin. After presenting the historical context of the events surrounding the implementation of Bitcoin and its distributed ledger that we know of as the blockchain, we intend to address the following questions that have come to us throughout the course of our work on this analysis:

* If Bitcoin provides an outlet for the poor to participate in the global economy by enabling people in these communities to make transactions on a blockchain without policy constraints, or trade barriers, to allow those in secluded communities to access a larger global economic network using these tools to conduct trade with digital assets, then what types of solutions can be built to enable the practical execution of these technologies on a local level?

* Can we explain the historical context of Bitcoin's adoption, and its emerging status as an asset class, when juxtaposed against the time-series data of a basket of global assets and Bitcoin network and Google Search Trends data, to test against the claim that Bitcoin is the new "Gold Standard" in a perpetually expanding digital economy?

* As Bitcoin gets closer to the validation of the final reward block to be mined, and the network stops rewarding miners for validating the transactions on the blockchain with Bitcoin, will the price decrease and force miners to increase the transaction fees associated to mining, which can lead to parity in terms of the transaction costs associated with traditional payment networks?

* What will the machine learning model we define tell us about the price of Bitcoin when transaction costs are increased or decreased?

* What features and predictors in our model will impact the price of Bitcoin the most?

### Methods & Results Overview
The intent is to implement a Linear Regression Model to fit weighted features against a bias to determine the outcome of a label or respionse variable, in this case, the price of Bitcoin. During the preprocessing of the data, this research walks through the process of feature engineering and the model selection used to attribute all data to appropriates value from direct sources while succeeding in obtaining ~3,000 observations from the ~30 features that this research captures for the implementation of the models.

To integrate [GoogleTrends](https://trends.google.com/trends/explore?date=2009-01-03%202018-08-09&geo=US&q=bitcoin) search data into the feature set, an implemention of a Python script was used for this research to normalize the daily data obtained over a 9 year period to ensure the flexibility to work with a precise level of daily granularity when analyzing price data against the chosen proxy for market demand, search data.

A price distribution assisted in selecting the proper correlation method to apply. Price distributions were plotted to display the daily performance return in the price of Bitcoin also. *The result of this plot suggests that there are more poor performing, or losing days in the price of Bitcoin than there are days that will produce a positive financial gain for a short-term speculator*. Because of the extremely skewed data, in both dependent and independent variables, a Spearman's Rank correlation method implementation is applied after the *Standardization* of all regression inputs and the transformation of the feature set into *z-scores*.

A Spearman's Rank Correlation Matrix helped to better find a series of features that displayed signs of having a high correlation, or inverse correlation against the price of Bitcoin, to help determine those features with a probability of having an association to the impact of its price. The goal is that the final result will enlighten the research conclusions to help determine if the associations found in the data can provide clues for causation that would explain the value of Bitcoin that may align with the historical context provided by this research. The Spearman's Rank correlation was performed against the price of Bitcoin, and an initial test of p-values helped to determine which features had a significant correlation with its price. *On this initial iteration of testing, all of the independent variables proved to correlate to the price with statistical significance*.

An initial regression model was fit to the feature set to calculate the Variance Inflation Factors for each feature to help find and reduce any impacts caused by known multicollinearity effects that can be caused by the extreme distribution skew of the features available. *Testing the features using their p-values, provided sufficient evidence to warrant the rejection of the null hypothesis and the probability that the r$^{2}$ values could result in 0 for the features remaining*. 

Three iterations of significance testing and regression modeling resulted in the following statistically significant features:
* Bitcoin Users Number of Transactions per Day	
* Average Transaction Confirmation Time	
* Transaction Volume in USD	
* Market Demand (Google Search Proxy)
* Profit & Loss (Added as a Categorical Feature to help describe fitted data set)

The models resulted in a total **r$^{2}$ = 0.762** which translates int the coefficient of determination. A statistical measure of how well the regression line approximates the real data points. This result allows the statistician to determine how well the set of all features can describe the model. Upon the selection of the model's features assisted by the p-value and Variance Inflation Factor testing performed, the observations fitted against the features modeled by the Multiple Linear Regression were plotted in 5D & 6D visualizations to understand the impact of the data and its relationship to the price of Bitcoin.

### Conclusions
Pending

### Limitations
* Google Trends data is only available as Monthly data over longer time horizons. This research includes a method to programmatically normalize the trend data into Daily data points to better compare it against Daily price data and other features used in the model presented.

* There does not appear to be many inversely correlated features to the price of Bitcoin. This may indicate that some other unobserved/unknown variable is having an impact on all asset classes as investors search for higher yields in an inflationary economic environment.

* This research will avoid performing any transformations to better achieve a linear relationships with the response variable due to the complexities of working with the re-interpretation of coefficients because of the potential that it may lead to the overfitting of the data used in these tests when the model is trained to predict a specific outcome.$^{31}$
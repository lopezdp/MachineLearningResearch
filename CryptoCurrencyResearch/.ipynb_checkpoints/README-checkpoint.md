# A Machine Learning Model using TensorFlow to Describe the Market Price of Cryptocurrencies: The Case of Bitcoin.
### Authors
* **Author/Researcher:** David P Lopez
* **Faculty Advisor & Author:** Professor Eric Hernandez PhD *Candidate*
* **Miami Dade College STEM Research Institute**

## Abstract
### Background
Digital currencies have led to an evolution in how society thinks about the abstract concept of money. This paradigm shift in the recording of transactions, and what it means to make a transaction with a secure store of value, as a secure and private medium of exchange, that is also enabled with a distributed and immutable general ledger, in a digital economy riddled with persistent threats against the frictionless flow of digital payments, is what we as the authors of this research, are attempting to quantify and discuss in order to better understand why and how a digital currency like Bitcoin, obtains the inherent value and prices that our society has observed in its short history. 

### Objectives
The objective of this body of work is to leverage the Python programming language and Open Source Machine Learning Libraries like Google's TensorFlow, to find features within a specific data set related to Bitcoin and other correlating asset classes, that we can observe and use to build a model using linear regression techniques to find a set of features that have a high degree of impact on the price of Bitcoin. After presenting the historical context of the events surrounding the implementation of Bitcoin and its distributed ledger that we know of as the blockchain, we intend to address the following questions that have come to us throughout the course of our work on this analysis:

* If Bitcoin provides an outlet for the poor to participate in the global economy by enabling people in these communities to make transactions on a blockchain without policy constraints, or trade barriers, to allow those in secluded communities to access a larger global economic network using these tools to conduct trade with digital assets, then what types of solutions can be built to enable the practical execution of these technologies on a local level?

* Can we explain the historical context of Bitcoin's adoption, and its emerging status as an asset class, when juxtaposed against the time-series data of a basket of global assets and Bitcoin network and Google Search Trends data, to test against the claim that Bitcoin is the new "Gold Standard" in a perpetually expanding digital economy?

* As Bitcoin gets closer to the validation of the final reward block to be mined, and the network stops rewarding miners for validating the transactions on the blockchain with Bitcoin, will the price decrease and force miners to increase the transaction fees associated to mining, which can lead to parity in terms of the transaction costs associated with traditional payment networks?

* What will the machine learning model we define tell us about the price of Bitcoin when transaction costs are increased or decreased?

* What features and predictors in our model will impact the price of Bitcoin the most?

### Methods
We intend to implement a Linear Regression Model to fit weighted features against a bias to determine the outcome of a label or target, in this case, the price of Bitcoin. During the preprocessing of our data, we ensured that all data was attributed with an appropriate value from a direct source and we succeeded in obtaining ~3,000 observations from the ~30 features that we were able to capture.

To integrate [GoogleTrends](https://trends.google.com/trends/explore?date=2009-01-03%202018-08-09&geo=US&q=bitcoin) search data into our feature set, we implemented a Python script to normalize the daily data that we obtained over a 9 year period to ensure that we were able to work with a precise level of daily granularity when analyzing price data against our proxy for market demand, search data.

We proceeded to implement a Pearson Correlation Matrix in order to find a series of features that showed signs of having a high correlation, or inverse correlation against the price of Bitcoin. For those features that showed a positive correlation to the price of Bitcoin, we arbitratily chose features which had a Pearson Correlation Value of >= 0.70. Finding an asset with a negative correlation proved to be more difficult. We arbitrarily chose inversely correlated features which had a Pearson Correlation Value of <= 0.15 as those displayed the lowest values in the matrix.

From the selection of correlated features, we found that the variability of units amongst the features required that we implement a proccess to standardize the regression inputs of the model to avoid any multicollinearity of features in the set. After standardizing and transforming the inputs into z-scores, we completed the implementation of our Multiple Linear Regression using a weighted sum of predictors to describe the output label.

Next, we calculated the *Variance Inflation Factors* of the set of all highly correlated features to find the ratio of variance between them and avoid the severity of any multicollinearity potentially found in the set and plotted a 3D hyperplane regression with the features that had a *Variance Inflation Factor* Output <= 5. We also plotted the partial regression for each of the features in the set against the price of Bitcoin.

Finally we implemented a Multiple Linear Regression model using the TensorFlow Machine Learning library. Using the same feature set of highly correlated predictors we trained the model using a learning rate of 0.35 in combination with a Gradient Descent Optimizer. Using the trained models in Seaborn, statsmodels, and TensorFlow, we proceeded to conduct tests against each feature to determine which feature, or predictor, showed the highest degree of statistical significance in terms of its impact on the price of Bitcoin to conclude any future outcomes that **may** occur under changing market conditions.

### Results

The data shows that the highest clusters of observations are concentrated at the intersection of the highest prices of Bitcoin, which correspond to a transaction cost of under 1.0%, as the gold market hovers just below a mean price of approximately $1350.

### Conclusions
Pending

### Limitations
* Google Trends data is only available as Monthly data over longer time horizons. This research includes a method to programmatically normalize the trend data into Daily data points to better compare it against Daily price data and other features used in the model presented.

* There does not appear to be many inversely correlated features to the price of Bitcoin. This may indicate that some other unobserved/unknown variable is having an impact on all asset classes as investors search for higher yields in an inflationary economic environment.
import numpy
import pandas
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pandas.read_csv("data\o-ring-erosion-onlyData", header = None)

data.columns = ["Number of O-rings at risk on a given flight", "Number experiencing thermal distress", "Launch temperature (degrees F)", "Leak-check pressure (psi)", "Temporal order of flight"]

data

data.drop(["Number of O-rings at risk on a given flight", "Temporal order of flight"], axis = 1, inplace = True)

data

data.describe()

cor = data.corr()
sb.heatmap(cor, annot = True, cmap = "coolwarm")

plt.figure(figsize = (12,5))

sb.distplot(data["Launch temperature (degrees F)"][data["Number experiencing thermal distress"] == 0], label = "No Thermal Distress")

sb.distplot(data["Launch temperature (degrees F)"][data["Number experiencing thermal distress"] != 0], label = "Thermal Distress Present")

plt.legend()
plt.show()

xData = data.drop(["Number experiencing thermal distress"], axis = 1)

yData = data["Number experiencing thermal distress"]

xData
yData

# Use sklearn and create a LinearRegression Object
alg = LinearRegression()

# Train the algorithm
alg.fit(xData, yData)

# Make prediction
alg.predict(numpy.array([31,150]).reshape(1,2))

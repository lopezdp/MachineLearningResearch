import pandas
import numpy

data = pandas.read_csv(r"C:\Users\David\OneDrive\Documents\DevOps\ML\Churn_Modeling.csv")

print(data.shape)

data.drop(['RowNumber','CustomerId','Surname'], axis = 1, inplace = True)

summary = data.describe()

data.isnull().sum()
data.Age[data.Exited == 1].count()

#Germany
gt = data.Geography[data.Geography == 'Germany'].count()
gl = data.Geography[data.Geography == 'Germany'][data.Exited == 1].count()

#Spain
st = data.Geography[data.Geography == 'Spain'].count()
sl = data.Geography[data.Geography == 'Spain'][data.Exited == 1].count()

#France
ft = data.Geography[data.Geography == 'France'].count()
fl = data.Geography[data.Geography == 'France'][data.Exited == 1].count()

data.Age.replace(numpy.nan,data.Age.mean(),inplace=True)

data.Age.var()
data['Age'].var()

#################
#dataViz
import matplotlib.pyplot as pyplot

x = numpy.arange(10,20,0,1)
y = numpy.sin(x)*4+10
z = 2*x+0.5*y

plt.figure(figsize = (12,5))
plt.plot(x,y,'r',label='x vs y')
plt.plot(x,z,'b',label='x vs z')
plt.title("This is my graph")
plt.xlabel("value of x")
plt.ylabel("value of y")
plt.show()

plt.figure(figsize=(12,5))
plt.scatter(x,y)

countries = data.Geography.unique()
t = [ft,st,gt]
l = [fl,sl,gl]

plt.figure(figsize = (12,8))
plt.pie(t,labels=countries)
plt.show()
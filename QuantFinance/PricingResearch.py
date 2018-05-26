import quandl as ql
import math
from IPython import display
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from pytrends.request import TrendReq

# Configure verbosity
tf.logging.set_verbosity(tf.logging.ERROR)

# Configure quandl API Key
# & Authenticate
apiKey = "-rLjiPduuzgzKp99MMHb"
ql.ApiConfig.api_key = apiKey    

# Capture historical BTC price data
# Data showing the USD market price from Mt.gox
price = ql.get("BCHAIN/MKPRU")
price = price.rename(columns={"Value": "Price"})

price.plot(title='BTC Market Price', figsize=(18,10))

# Bitcoin Difficulty
# Difficulty is a measure of how difficult it is to find a hash below a given target.
diff = ql.get("BCHAIN/DIFF")
diff = diff.rename(columns={"Value": "Network Difficulty"})

diff.plot(title='BTC Difficulty', figsize=(18,10))

# Bitcoin Average Block Size
# The Average block size in MB
avbls = ql.get("BCHAIN/AVBLS")
avbls = avbls.rename(columns={"Value": "Average Block Size"})

avbls.plot(title='BTC Avg. Block Size', figsize=(18,10))

# Bitcoin Median Transaction Confirmation Time
# The Dailiy Median time take for transactions to be accepted into a block.
atrct = ql.get("BCHAIN/ATRCT")
atrct = atrct.rename(columns={"Value": "Median Confirmation Time"})

atrct.plot(title='BTC Median Confirmation Time', figsize=(18,10))

# Bitcoin Hash Rate
# The estimated number of giga hashes per second 
# (billions of hashes per second) the bitcoin network is performing.
hrate = ql.get("BCHAIN/HRATE")
hrate = hrate.rename(columns={"Value": "Network Hash Rate"})

hrate.plot(title='BTC Network Hash Rate', figsize=(18,10), logy=True)

# Bitcoin Cost % of Transaction Volume
# Data showing miners revenue as as percentage of the transaction volume.
cptrv = ql.get("BCHAIN/CPTRV")
cptrv = cptrv.rename(columns={"Value": "Cost % of Transaction Volume"})

cptrv.plot(title='BTC Cost % of Transaction Volume', figsize=(18,10), logy=True)

# Bitcoin Estimated Transaction Volume
# Similar to the total output volume with the addition of an algorithm 
# which attempts to remove change from the total value. This may be a more 
# accurate reflection of the true transaction volume.
etrav = ql.get("BCHAIN/ETRAV")
etrav = etrav.rename(columns={"Value": "Est. Transaction Volume"})

etrav.plot(title='BTC Est. Transaction Volume', figsize=(18,10), logy=True)

# Bitcoin Total Output Volume
# The total value of all transaction outputs per day. This includes coins 
# which were returned to the sender as change.
toutv = ql.get("BCHAIN/TOUTV")
toutv = toutv.rename(columns={"Value": "Total Output Volume"})

toutv.plot(title='BTC Total Output Volume', figsize=(18,10), logy=True)

# Bitcoin Number of Transactions per Block
# The average number of transactions per block.
ntrbl = ql.get("BCHAIN/NTRBL")
ntrbl = ntrbl.rename(columns={"Value": "Number of Transactions per Block"})

ntrbl.plot(title='BTC Number of Transactions per Block', figsize=(18,10), logy=True
           
# Bitcoin Number of Unique Bitcoin Addresses Used
# Number of unique bitcoin addresses used per day.
naddu = ql.get("BCHAIN/NADDU")
naddu = naddu.rename(columns={"Value": "Number of Unique Bitcoin Addresses Used"})

naddu.plot(title='BTC Number of Unique Bitcoin Addresses Used', figsize=(18,10), logy=True)
           
# Bitcoin Total Number of Transactions
# Total number of unique bitcoin transactions per day.
ntrat = ql.get("BCHAIN/NTRAT")
ntrat = ntrat.rename(columns={"Value": "Total Number of Transactions"})

ntrat.plot(title='BTC Total Number of Transactions', figsize=(18,10), logy=True)
           
# Bitcoin Number of Transactions
# Total number of unique bitcoin transactions per day.
ntran = ql.get("BCHAIN/NTRAN")
ntran = ntran.rename(columns={"Value": "Number of Transactions"})

ntran.plot(title='BTC Number of Transactions', figsize=(18,10), logy=True)
           
# Bitcoin Total Transaction Fees
# Data showing the total BTC value of transaction fees miners earn per day.
trfee = ql.get("BCHAIN/TRFEE")
trfee = trfee.rename(columns={"Value": "Total Transaction Fees"})

trfee.plot(title='BTC Total Transaction Fees', figsize=(18,10), logy=True)

# Total Bitcoins
# Data showing the historical total number of bitcoins which have been mined.
totbc = ql.get("BCHAIN/TOTBC")
totbc = totbc.rename(columns={"Value": "Total Bitcoins"})

totbc.plot(title='BTC Total Bitcoins', figsize=(18,10), logy=True)

# Bitcoin Miners Revenue
# Historical data showing: 
# (number of bitcoins mined per day + transaction fees) * market price.
mirev = ql.get("BCHAIN/MIREV")
mirev = mirev.rename(columns={"Value": "Miners Revenue"})

mirev.plot(title='BTC Miners Revenue', figsize=(18,10), logy=True)
           
# Bitcoin Cost Per Transaction
# Data showing miners revenue divided by the number of transactions.
cptra = ql.get("BCHAIN/CPTRA")
cptra = cptra.rename(columns={"Value": "Cost Per Transaction"})

cptra.plot(title='BTC Cost Per Transaction', figsize=(18,10), logy=True)

# Bitcoin USD Exchange Trade Volume
# Data showing the USD trade volume from the top exchanges.
trvou = ql.get("BCHAIN/TRVOU")
trvou = trvou.rename(columns={"Value": "USD Exchange Trade Volume"})

trvou.plot(title='USD Exchange Trade Volume', figsize=(18,10), logy=True)

# Bitcoin Estimated Transaction Volume USD
# Similar to the total output volume with the addition of an algorithm 
# which attempts to remove change from the total value. This may be a 
# more accurate reflection of the true transaction volume.
etrvu = ql.get("BCHAIN/ETRVU")
etrvu = etrvu.rename(columns={"Value": "Est. Transaction Volume USD"})

etrvu.plot(title='Est. Transaction Volume USD', figsize=(18,10), logy=True)

# Bitcoin Total Transaction Fees USD
# Data showing the total BTC value of transaction fees miners earn per day in USD.
trfus = ql.get("BCHAIN/TRFUS")
trfus = trfus.rename(columns={"Value": "Total Transaction Fees USD"})

trfus.plot(title='Total Transaction Fees USD', figsize=(18,10), logy=True)

# Bitcoin Market Capitalization
# Data showing the total number of bitcoins in circulation at the market price in USD.
mktcp = ql.get("BCHAIN/MKTCP")
mktcp = mktcp.rename(columns={"Value": "Market Capitalization"})

mktcp.plot(title='BTC Market Capitalization', figsize=(18,10), logy=True)

# Bitcoin api.blockchain Size
# The total size of all block headers and transactions. 
# Not including database indexes.
blchs = ql.get("BCHAIN/BLCHS")
blchs = blchs.rename(columns={"Value": "api.blockchain Size"})

blchs.plot(title='BTC api.blockchain Size', figsize=(18,10), logy=True)
           
# Bitcoin My Wallet Number of Transaction Per Day
# Number of transactions made by MyWallet Users per day.
mwntd = ql.get("BCHAIN/MWNTD")
mwntd = mwntd.rename(columns={"Value": "BTC Wallet Transactions Per Day"})

mwntd.plot(title='BTC Wallet Transactions Per Day', figsize=(18,10), logy=True)
           
# Bitcoin My Wallet Number of Users
# Number of wallets hosts using our MyWallet Service.
mwnus = ql.get("BCHAIN/MWNUS")
mwnus = mwnus.rename(columns={"Value": "BTC Wallet Number of Users"})

mwnus.plot(title='BTC Wallet Number of Users', figsize=(18,10), logy=True)
           
# Bitcoin My Wallet Transaction Volume
# 24hr Transaction Volume of our MyWallet service.
mwtrv = ql.get("BCHAIN/MWTRV")
mwtrv = mwtrv.rename(columns={"Value": "BTC Wallet Transaction Volume"})

mwtrv.plot(title='BTC Wallet Transaction Volume', figsize=(18,10), logy=True)
           
# Data Currently Not Used
#############################################
# Bitcoin Days Destroyed (Minimum Age 1 Year)
bcddy = ql.get("BCHAIN/BCDDY")
bcddy = bcddy.rename(columns={"Value": "Days Destroyed (Minimum Age 1 Year)"})

bcddy.plot(title='BTC Days Destroyed (Minimum Age 1 Year', figsize=(18,10), logy=True)        
           
# Bitcoin Days Destroyed (Minimum Age 1 Month)
bcddm = ql.get("BCHAIN/BCDDM")
bcddm = bcddm.rename(columns={"Value": "Days Destroyed (Minimum Age 1 Month)"})

bcddm.plot(title='BTC Days Destroyed (Minimum Age 1 Month', figsize=(18,10), logy=True)
           
# Bitcoin Days Destroyed (Minimum Age 1 Week)
bcddw = ql.get("BCHAIN/BCDDW")
bcddw = bcddw.rename(columns={"Value": "Days Destroyed (Minimum Age 1 Week)"})

bcddw.plot(title='BTC Days Destroyed (Minimum Age 1 Week', figsize=(18,10), logy=True)
           
# Bitcoin Days Destroyed
bcdde = ql.get("BCHAIN/BCDDE")
bcdde = bcdde.rename(columns={"Value": "Days Destroyed"})

bcdde.plot(title='BTC Days Destroyed', figsize=(18,10), logy=True)
           
# Bitcoin Days Destroyed Cumulative
bcddc = ql.get("BCHAIN/BCDDC")
bcddc = bcddc.rename(columns={"Value": "Days Destroyed Cumulative"})

bcddc.plot(title='BTC Days Destroyed Cumulative', figsize=(18,10), logy=True)          
           
# Bitcoin Trade Volume vs Transaction Volume Ratio
# Data showing the relationship between BTC transaction volume and USD exchange volume.
tvtvr = ql.get("BCHAIN/TVTVR")
tvtvr = tvtvr.rename(columns={"Value": "Trade Volume vs Transaction Volume Ratio"})

tvtvr.plot(title='BTC Trade Volume vs Transaction Volume Ratio', figsize=(18,10), logy=True)
           
# Bitcoin Network Deficit
# Data showing difference between transaction fees and cost of bitcoin mining.
netdf = ql.get("BCHAIN/NETDF")
netdf = netdf.rename(columns={"Value": "Network Deficit"})

netdf.plot(title='BTC Network Deficit', figsize=(18,10))
           
# Bitcoin Mining Operating Margin
# Data showing miners revenue minus estimated electricity and bandwidth costs.
miopm = ql.get("BCHAIN/MIOPM")
miopm = miopm.rename(columns={"Value": "Mining Operating Margin"})

miopm.plot(title='BTC Mining Operating Margin', figsize=(18,10), logy=True)
           
# testing the data structure and its values
frames = [price, diff, mwntd, mwnus, mwtrv, avbls, blchs,
          atrct, hrate, cptrv, etrav, toutv, ntrbl, naddu,
          ntrep, ntrat, ntran, trfee, totbc, mirev, cptra,
          trvou, etrvu, trfus, mktcp]
#, bcddy, bcddm, bcddw, bcdde, bcddc, tvtvr, netdf, miopm]
           
btcData = pd.concat(frames, axis=1)
btcData.shape           
         
btcData.describe()
           
btcCorr = btcData.corr()

plt.figure(figsize = (33,33))
sb.heatmap(btcCorr, annot = True, linewidths = .5, cmap = "coolwarm")

# Create a Google Trend Object
pytrends = TrendReq(hl='en-US', tz=360)

# Declare a var to store the search term
kw_list = ["bitcoin"]

# Build payload request to get data from Google trends
pytrends.build_payload(kw_list, cat=0, timeframe='2009-01-03 2018-05-26', geo='', gprop='')

# Get interest over time
trend = pytrends.interest_over_time()

# Plot the Interest
trend.plot(figsize=(20,10))
           
trend.shape

# Capture historical BTC price data
# Data showing the USD market price from Mt.gox
# Testing time-series vs pd.DF
           
price1 = ql.get("BCHAIN/MKPRU", collapse="weekly", returns="numpy")
           
#price1 = price1.rename(columns={"Value": "Price"})
           
#price1.plot(title='BTC Market Price', figsize=(18,10))   

price1
      
trend = trend.values

trend[105][0]           

price1[440][1]
           
price1[1150][0]

price1.shape


           
price1 = price1.drop(['Date'], 1)

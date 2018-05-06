import quandl as ql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

# Configure quandl API Key
# & Authenticate
apiKey = "-rLjiPduuzgzKp99MMHb"
ql.ApiConfig.api_key = apiKey    

# Capture historical BTC price data
price = ql.get("BCHAIN/MKPRU")
price = price.rename(columns={"Value": "Price"})

# Bitcoin Difficulty
diff = ql.get("BCHAIN/DIFF")
diff = diff.rename(columns={"Value": "Network Difficulty"})

# Bitcoin My Wallet Number of Transaction Per Day
mwntd = ql.get("BCHAIN/MWNTD")
mwntd = mwntd.rename(columns={"Value": "BTC Wallet Transactions Per Day"})


# Bitcoin My Wallet Number of Users
mwnus = ql.get("BCHAIN/MWNUS")
mwnus = mwnus.rename(columns={"Value": "BTC Wallet Number of Users"})

# Bitcoin My Wallet Transaction Volume
mwtrv = ql.get("BCHAIN/MWTRV")
mwtrv = mwtrv.rename(columns={"Value": "BTC Wallet Transaction Volume"})

# Bitcoin Average Block Size
avbls = ql.get("BCHAIN/AVBLS")
avbls = avbls.rename(columns={"Value": "Average Block Size"})

# Bitcoin api.blockchain Size
blchs = ql.get("BCHAIN/BLCHS")
blchs = blchs.rename(columns={"Value": "api.blockchain Size"})

# Bitcoin Median Transaction Confirmation Time
atrct = ql.get("BCHAIN/ATRCT")
atrct = atrct.rename(columns={"Value": "Median Confirmation Time"})

# Bitcoin Hash Rate
hrate = ql.get("BCHAIN/HRATE")
hrate = hrate.rename(columns={"Value": "Network Hash Rate"})

# Bitcoin Cost % of Transaction Volume
cptrv = ql.get("BCHAIN/CPTRV")
cptrv = cptrv.rename(columns={"Value": "Cost % of Transaction Volume"})

# Bitcoin Estimated Transaction Volume
etrav = ql.get("BCHAIN/ETRAV")
etrav = etrav.rename(columns={"Value": "Est. Transaction Volume"})

# Bitcoin Total Output Volume
toutv = ql.get("BCHAIN/TOUTV")
toutv = toutv.rename(columns={"Value": "Total Output Volume"})

# Bitcoin Number of Transaction per Block
ntrbl = ql.get("BCHAIN/NTRBL")
ntrbl = ntrbl.rename(columns={"Value": "Number of Transaction per Block"})

# Bitcoin Number of Unique Bitcoin Addresses Used
naddu = ql.get("BCHAIN/NADDU")
naddu = naddu.rename(columns={"Value": "Number of Unique Bitcoin Addresses Used"})

# Bitcoin Number of Transactions Excluding Popular Addresses
ntrep = ql.get("BCHAIN/NTREP")
ntrep = ntrep.rename(columns={"Value": "Number of Transactions Excluding Popular Addresses"})

# Bitcoin Total Number of Transactions
ntrat = ql.get("BCHAIN/NTRAT")
ntrat = ntrat.rename(columns={"Value": "Total Number of Transactions"})

# Bitcoin Number of Transactions
ntran = ql.get("BCHAIN/NTRAN")
ntran = ntran.rename(columns={"Value": "Number of Transactions"})

# Bitcoin Total Transaction Fees
trfee = ql.get("BCHAIN/TRFEE")
trfee = trfee.rename(columns={"Value": "Total Transaction Fees"})

# Total Bitcoins
totbc = ql.get("BCHAIN/TOTBC")
totbc = totbc.rename(columns={"Value": "Total Bitcoins"})

# Bitcoin Miners Revenue
mirev = ql.get("BCHAIN/MIREV")
mirev = mirev.rename(columns={"Value": "Miners Revenue"})

# Bitcoin Cost Per Transaction
cptra = ql.get("BCHAIN/CPTRA")
cptra = cptra.rename(columns={"Value": "Cost Per Transaction"})

# Bitcoin USD Exchange Trade Volume
trvou = ql.get("BCHAIN/TRVOU")
trvou = trvou.rename(columns={"Value": "USD Exchange Trade Volume"})

# Bitcoin Estimated Transaction Volume USD
etrvu = ql.get("BCHAIN/ETRVU")
etrvu = etrvu.rename(columns={"Value": "Est. Transaction Volume USD"})

# Bitcoin Total Transaction Fees USD
trfus = ql.get("BCHAIN/TRFUS")
trfus = trfus.rename(columns={"Value": "Total Transaction Fees USD"})

# Bitcoin Market Capitalization
mktcp = ql.get("BCHAIN/MKTCP")
mktcp = mktcp.rename(columns={"Value": "Market Capitalization"})

# Bitcoin Days Destroyed (Minimum Age 1 Year)
bcddy = ql.get("BCHAIN/BCDDY")
bcddy = bcddy.rename(columns={"Value": "Days Destroyed (Minimum Age 1 Year)"})

# Bitcoin Days Destroyed (Minimum Age 1 Month)
bcddm = ql.get("BCHAIN/BCDDM")
bcddm = bcddm.rename(columns={"Value": "Days Destroyed (Minimum Age 1 Month)"})

# Bitcoin Days Destroyed (Minimum Age 1 Week)
bcddw = ql.get("BCHAIN/BCDDW")
bcddw = bcddw.rename(columns={"Value": "Days Destroyed (Minimum Age 1 Week)"})

# Bitcoin Days Destroyed
bcdde = ql.get("BCHAIN/BCDDE")
bcdde = bcdde.rename(columns={"Value": "Days Destroyed"})

# Bitcoin Days Destroyed Cumulative
bcddc = ql.get("BCHAIN/BCDDC")
bcddc = bcddc.rename(columns={"Value": "Days Destroyed Cumulative"})

# Bitcoin Trade Volume vs Transaction Volume Ratio
tvtvr = ql.get("BCHAIN/TVTVR")
tvtvr = tvtvr.rename(columns={"Value": "Trade Volume vs Transaction Volume Ratio"})

# Bitcoin Network Deficit
netdf = ql.get("BCHAIN/NETDF")
netdf = netdf.rename(columns={"Value": "Network Deficit"})

# Bitcoin Mining Operating Margin
miopm = ql.get("BCHAIN/MIOPM")
miopm = miopm.rename(columns={"Value": "Mining Operating Margin"})

# testing the data structure and its values
frames = [price, diff, mwntd, mwnus, mwtrv, avbls, blchs,
          atrct, hrate, cptrv, etrav, toutv, ntrbl, naddu,
          ntrep, ntrat, ntran, trfee, totbc, mirev, cptra,
          trvou, etrvu, trfus, mktcp]
#, bcddy, bcddm, bcddw,
#          bcdde, bcddc, tvtvr, netdf, miopm]

btcData = pd.concat(frames, axis=1) 
btcData

# Create a Google Trend Object
pytrends = TrendReq(hl='en-US', tz=360)

# Declare a var to store the search term
kw_list = ["bitcoin"]

# Build payload request to get data from Google trends
pytrends.build_payload(kw_list, cat=0, timeframe='2009-01-03 2018-04-27', geo='', gprop='')

# Get interest over time
pytrends.interest_over_time()

# Plot the Interest
pytrends.interest_over_time().plot(figsize=(20,10))

##############################################
# Create a Google Trend Object
pytrends1 = TrendReq(hl='en-US', tz=360)

# Declare a var to store the search term
kw_list = ["bitcoin"]

# Build payload request to get data from Google trends
pytrends1.build_payload(kw_list, cat=0, timeframe='2009-01-13 2009-01-20', geo='', gprop='')

# Get interest over time
pytrends1.interest_over_time()









# Need to merge these column values according to date






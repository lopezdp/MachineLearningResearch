import quandl
import pandas
import numpy

# Configure quandl API Key
# & Authenticate
apiKey = "-rLjiPduuzgzKp99MMHb"
quandl.ApiConfig.api_key = apiKey    

# Capture historical BTC price data
btc = quandl.get("BCHAIN/MKPRU")

# Bitcoin Difficulty
diff = quandl.get("BCHAIN/DIFF")

# Bitcoin My Wallet Number of Transaction Per Day
mwntd = quandl.get("BCHAIN/MWNTD")

# Bitcoin My Wallet Number of Users
mwnus = quandl.get("BCHAIN/MWNUS")

# Bitcoin My Wallet Transaction Volume
mwtrv = quandl.get("BCHAIN/MWTRV")

# Bitcoin Average Block Size
avbls = quandl.get("BCHAIN/AVBLS")

# Bitcoin api.blockchain Size
blchs = quandl.get("BCHAIN/BLCHS")

# Bitcoin Median Transaction Confirmation Time
atrct = quandl.get("BCHAIN/ATRCT")

# Bitcoin Hash Rate
hrate = quandl.get("BCHAIN/HRATE")

# Bitcoin Cost % of Transaction Volume
cptrv = quandl.get("BCHAIN/CPTRV")

# Bitcoin Estimated Transaction Volume
etrav = quandl.get("BCHAIN/ETRAV")

# Bitcoin Total Output Volume
toutv = quandl.get("BCHAIN/TOUTV")

# Bitcoin Number of Transaction per Block
ntrbl = quandl.get("BCHAIN/NTRBL")

# Bitcoin Number of Unique Bitcoin Addresses Used
naddu = quandl.get("BCHAIN/NADDU")

# Bitcoin Number of Transactions Excluding Popular Addresses
ntrep = quandl.get("BCHAIN/NTREP")

# Bitcoin Total Number of Transactions
ntrat = quandl.get("BCHAIN/NTRAT")

# Bitcoin Number of Transactions
ntran = quandl.get("BCHAIN/NTRAN")

# Bitcoin Total Transaction Fees
trfee = quandl.get("BCHAIN/TRFEE")

# Total Bitcoins
totbc = quandl.get("BCHAIN/TOTBC")

# Bitcoin Miners Revenue
mirev = quandl.get("BCHAIN/MIREV")




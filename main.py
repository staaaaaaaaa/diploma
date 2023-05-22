from yahooquery import Screener

s = Screener()

# data is a dictionary containing the keys passed to the function
data = s.get_screeners('utilities', count=25)

# the majority of the data will be in the quotes key
print(data['ms_technology']['quotes'][1])
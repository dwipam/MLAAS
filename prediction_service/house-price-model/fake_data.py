import numpy as np
import pandas as pd

N = 4000
sqft = np.random.randint(300,3000,size=N)
garden_area = np.random.randint(300,8000,size=N)
white_noise = np.random.normal(loc=0.,scale=1.,size=N)
price = 40*sqft + \
		60*garden_area + \
		white_noise/10.

data = pd.DataFrame({"price" : price, "sqft" : sqft, "garden_area" : garden_area})
print data.describe()

train = data.sample(frac=0.8)
test = data[~data.index.isin(train.index)]

train.to_csv("train.csv")
test.to_csv("test.csv")

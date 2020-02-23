import pandas as pd
import statsmodels.api as sm
import math

homeData = pd.read_csv("data/HousingPrice.csv")

nan_cols = [i for i in homeData.columns if homeData[i].isnull().any()]
print("Columns with missing data")
print(nan_cols)

print("\nQuestion 8.1")
print("Frequency Table for EXTERQUAL: \n")
eqFreqTable = pd.crosstab(index=homeData["ExterQual"], columns="Freq")
eqFreqTable['FreqPer'] = eqFreqTable/eqFreqTable.sum()
print(eqFreqTable)

print('\nQuestion 8.3')
print("Create Dummy variable for ExterQual so that TA = 1, zero otherwise\n")
dummies = pd.get_dummies(homeData.ExterQual, prefix='ExterQual')
dummies = dummies['ExterQual_TA']
homeData = pd.concat([homeData, dummies], axis=1)
print(homeData.head())

print('\nQuestion 8.4 and Question 8.6')
print('Logistic Regression and Confidence Interval')
# y = ExterQual_TA, x = SalePrice
salePrice = sm.add_constant(homeData['SalePrice'])
model = sm.Logit(list(homeData['ExterQual_TA']), salePrice)
result = model.fit()
print(result.summary())

print('\nQuestion 8.5')
logOdds = 5.8398 - .00002961 * 200000
print(logOdds)
odds = math.exp(-.08)
print(odds)

print('\nQuestion 8.7')
xValues = homeData[['SalePrice', 'YearBuilt']]
xValues = sm.add_constant(xValues)
model2 = sm.Logit(list(homeData['ExterQual_TA']), xValues)
result2 = model2.fit()
print(result2.summary())

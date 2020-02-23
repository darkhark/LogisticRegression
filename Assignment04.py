import pandas as pd
homeData = pd.read_csv("data/HousingPrice.csv")
print("Question 8.1")
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



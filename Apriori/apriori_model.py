# load libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from apyori import apriori


# load dataset
store_data = pd.read_csv('G:/rauf/STEPBYSTEP/Projects/ML/Machine Learning/Apriori/data/store_data.csv', header=None)
#print(store_data.head())


# our apriori algorithm intake data as list of lists which whole data is one list and each transactions sublist in one list
# to do this we have to convert panda dataframe to list
records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])


# define apriori algorithm 
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2) # here some parameters defined as treshold case we do not want make calculation on whole dataset
association_results = list(association_rules)


# lets see how many product bout together
print(len(association_rules)) # here we go 48 rules found


# lets see which product is sold most time
print(association_rules[0])
# out ----> RelationRecord(items=frozenset({'light cream', 'chicken'}), support=0.004532728969470737, ordered_statistics[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)])
# from result you can understand everything
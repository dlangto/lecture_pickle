import pandas
from sklearn.ensemble import RandomForestClassifier
import pickle
dataset = pandas.read_csv("private_dataset.csv")
target = dataset.iloc[:,30].values
#target is all the rows in the 31st column in the dataset. Values conversts it into something sklearn can read
data = dataset. iloc[:,0:30]
#0 is inclusive, 30 is not inclusive. Columns 1-29 are all our data

machine = RandomForestClassifier(criterion="gini", max_depth=10, n_estimators=11)
machine.fit(data, target)

with open("machine.pickle","wb") as file:
	pickle.dump(machine, file)
#make and open a pickle file with the writing format as binary, not text- that is wb
#if you want to send it to someone, send them machine.pickle. This sends them the information encripted
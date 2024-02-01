#we use new_survey, which has x variables, but no y variables. Use the machine to get it
import pandas
import pickle

with open("machine.pickle", "rb") as file:
	machine = pickle.load(file)
	#use pickle to open the machine as a binary reader, make it the machine
new_survey = pandas.read_csv("new_survey.csv")
new_survey = new_survey.values
print(machine.predict(new_survey))

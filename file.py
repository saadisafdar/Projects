import pandas as pd

df = pd.read_csv("patients.csv")

print(df)
    
high_health = df[df['health_index'] > 92]

high_cost = df[df['treatment_cost'] > 100000]

print("Patients with highest health index:\n ", high_health['ID'], high_health['health_index'])

print("Patients with highest treatment cost:\n ",high_cost['ID'])
 
import numpy as np

heartbeat_readings = [72, 78, 75, 82, 88, 85, 92]

difference = np.diff(heartbeat_readings)


print("Name: ", " Saadi Safdar, ", " DOB: ", "4/2/2006")
print("Father name: ", " Muhammad Safdar Butt, ", " DOB: ", "31/8/1978")
print("Name: ", " Muhammad Siddique Butt, ", " DOB: ", "18/5/1960")
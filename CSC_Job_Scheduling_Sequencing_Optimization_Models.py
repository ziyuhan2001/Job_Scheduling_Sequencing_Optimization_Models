# -*- coding: utf-8 -*-
#Importing Scheduling Input File into Optimization Model
from google.colab import drive
import os
import pandas as pd
import datetime
from datetime import date, timedelta
import calendar
import numpy as np

drive.mount('/content/drive')

#Search Parameters setup
filename = 'PPbyLine_Output.xlsx'
search_path = '/content/drive/My Drive'

#Search for the File
for root, dirs, files in os.walk(search_path):
    if filename in files:
        file_path = os.path.join(root, filename)
        break

df = pd.read_excel(file_path, sheet_name=0)

#Date and Working Hours Calculation
def findDay(start_date):
    day = datetime.datetime.strptime(start_date, '%m/%d/%Y').weekday()
    return (day)

#Set Scheduling Start Date!
scheduling_start_date = '08/07/2023'
start_date = findDay(scheduling_start_date)

#Set Tonnage Capacity for each Shift! This determines the maximum tonnage capacity that each line can have during the 2-week scheduling period
#shift_ton_cap = 100
#total_ton_cap = shift_ton_cap * 26

#Total Working Hours by Day
Monday = 19
Tuesday = 19
Wednesday = 19
Thursday = 19
Friday = 17
Saturday_Day_Shift = 6
Saturday_Night_Shift = 6

#Working Hours associated with each day (from Day 0 to Day 11). Outputs each days working hours depending on what day it is today
Hours = [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday_Day_Shift + Saturday_Night_Shift, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday_Day_Shift + Saturday_Night_Shift]
Day_Hours = []

for i in range(6):
  Day_Hours.append(Hours[start_date+i])

for j in range(6):
  Day_Hours.append(Hours[start_date+j])

#Calculating working hours available for jobs until Critical Date deadline
def CriticalDateHours(day_index):
  critical_date_hours = sum(Day_Hours[:(day_index+1)])
  return(critical_date_hours)

#Data Cleansing

#Remove rows that have an empty/null depth value
df.dropna(subset = ['Max Depth (Inches)'], inplace = True)
df = df.reset_index(drop=True)

#Critical Date Index to indicate how many days apart the critical date is from the current date
start_date = datetime.datetime.strptime(scheduling_start_date, '%m/%d/%Y').date()
df['Critical Date'] = pd.to_datetime(df['Critical Date'])
df['Critical Date'] = df['Critical Date'].dt.date
df['Critical Date Index'] = (df['Critical Date'] - start_date) / np.timedelta64(1, 'D')
df.insert(2, 'Critical Date Index', df.pop('Critical Date Index'))
df['Critical Date Index'] = df['Critical Date Index'].astype('int')

#Priority for earlier critical dates for similar tonnage jobs
df['Duplicate'] = df['Tons'].map(df['Tons'].value_counts() > 1)
df['Priority'] = np.where(df['Duplicate'] == True, df['Critical Date Index'], 0)

#Critical Date Hours: converting Critical Date Index into Working Hours
df['Critical Date Hours'] = df['Critical Date Index'].apply(lambda x: CriticalDateHours(x))
df.insert(3, 'Critical Date Hours', df.pop('Critical Date Hours'))

#Time Available: indicates the range of time available to assign the job. If a job yields a negative Time Available value, change to 0 to schedule the job ASAP.
df['Time Available (Hours)'] = df['Critical Date Hours'] - df['Estimated Time (Hours)']
df.insert(4, 'Time Available (Hours)', df.pop('Time Available (Hours)'))
df['Time Available (Hours)'] = np.where(df['Time Available (Hours)'] < 0, 0, df['Time Available (Hours)'])

#Convert Max Depth and Max Length columns to integer column type
df['Max Depth (Inches)'] = df['Max Depth (Inches)'].astype('int')
df['Max Length (Feet)'] = df['Max Length (Feet)'].astype('int')

#Insert 'Production Line' Column to indicate the possible Production Line(s) a job requisition can go to
df['P1: Longspan'] = np.where((df['Max Depth (Inches)'] <= 120) & (df['Max Depth (Inches)'] >= 30) & (df['Max Length (Feet)'] <= 120) & (df['Max Length (Feet)'] > 62), 1, 0)
df['P2: Midspan'] = np.where((df['Max Depth (Inches)'] <= 36) & (df['Max Length (Feet)'] <= 62), 1, 0)
df['P4: Martignetti'] = np.where((df['Max Depth (Inches)'] <= 48) & (df['Max Length (Feet)'] <= 90), 1, 0)

#If Jobs are not suited to be put onto P1 or P2 line (does not P1 or P2 line criteria), immediately assign to P4 line
df['P4: Martignetti'] = np.where((df['P1: Longspan'] == 0) & (df['P2: Midspan'] == 0) & (df['P4: Martignetti'] == 0), 1, df['P4: Martignetti'])

#Data Inputs:

#Set of Jobs
Jobs = df["Requisition"].values.tolist()

#Set of Production Lines
Lines = ["P1", "P2", "P4"]

#Critical Date Index for each Job
CDI = df["Critical Date Index"].values.tolist()

#Days of the Scheduling Process (Assign each job to a date)
Days = ["Day 0", "Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10", "Day 11", "Day 12"]

#Total Available Production Time (2-week span)
Total_Hours = sum(Day_Hours[:])

#Tonnage Weight of each Job
weights = df["Tons"].values.tolist()

#Estimated Completion Time (Hours) of each Job
e_times = df["Estimated Time (Hours)"].values.tolist()

#Matrix (nested list) of Production Lines that a job can potentially go to
df1 = df[['P1: Longspan', 'P2: Midspan', 'P4: Martignetti']]
Possible_Lines = df1.values.tolist()

#--------------------------------------------------------------------------------------------
#Optimization Model: Optimal Production Line Assignment
#Install and import packages:
!pip install mip
from mip import *
import math

#Model:
m = Model(name='Scheduling Optimization Model', sense=MINIMIZE)

#Variables:
X = [[m.add_var(name = "X_({},{})".format(j,p) , var_type = BINARY) for p in range(len(Lines))] for j in range(len(Jobs))] #Decision Variable
T = [m.add_var(name = "T_({})".format(p)) for p in range(len(Lines))] #Total Tonnage Capacity at each Line
H = [m.add_var(name = "H_({})".format(p)) for p in range(len(Lines))] #Completion Time at each Line
D = [m.add_var(name = "D_({})".format(p)) for p in range(len(Lines)-1)] #Tonnage Difference across Lines

#Objective Function:
m.objective = xsum(D[p] for p in range(len(Lines)-1))

#Constraints:
#Constraint 1: Determining Tonnage Difference across all 3 lines
for p in range(len(Lines)-1):
  m += D[p] >= T[p+1] - T[p]
for p in range(len(Lines)-1):
  m += D[p] >= T[p] - T[p+1]

#Constraint 2: Determining tonnage capacity on each line
for p in range(len(Lines)):
  m += T[p] == xsum(weights[j]*X[j][p] for j in range(len(Jobs)))

#Constraint 3: Determining total production time on each line
for p in range(len(Lines)):
  m += H[p] == xsum(e_times[j]*X[j][p] for j in range(len(Jobs)))

#Constraint 4: Each job can only be assigned to one production line
for j in range(len(Jobs)):
  m += xsum(X[j][p] for p in range(len(Lines))) == 1

#Constraint 5: Each job can only be assigned to its possible production lines based on depth and length criteria
for j in range(len(Jobs)):
  for p in range(len(Lines)):
      m += X[j][p] <= Possible_Lines[j][p]

#--------------------------------------------------------------------------------------------
#Solve Optimization Model: Optimal Production Line Assignment
status = m.optimize()

if status == OptimizationStatus.OPTIMAL:
  print("Optimal Schedule")
  print("Objective Value:", m.objective_value)
  for j in range(len(Jobs)):
    for p in range(len(Lines)):
      if X[j][p].x == 1:
        print("Job Requisition: [{}] Line: [{}] --> Decision: {}".format(Jobs[j], Lines[p], X[j][p].x))

  for p in range(len(Lines)):
    if T[p].x > 0:
      print("Production Line {} --> Tonnage Capacity: {} (tons)".format(Lines[p], round(T[p].x,2)))

  for p in range(len(Lines)):
    if H[p].x > 0:
      print("Production Line {} --> Completion Time: {} (line hours)".format(Lines[p], round(H[p].x,2)))

elif status == OptimizationStatus.UNBOUNDED:
  print("Unbounded")

elif status == OptimizationStatus.INFEASIBLE:
  print("Infeasible")

else:
  print("ERROR, unexpected result")

#Jobs on Each Line
P1_All = []
P2_All = []
P4_All = []

#Creating a list for each production line to show the job requisitions assigned to each line
for i in range(len(Jobs)):
  if X[i][0].x == 1:
    P1_All.append(Jobs[i])

for j in range(len(Jobs)):
  if X[j][1].x == 1:
    P2_All.append(Jobs[j])

for k in range(len(Jobs)):
  if X[k][2].x == 1:
    P4_All.append(Jobs[k])

#Creating a dataframe for each production line, including only the job requisition and corresponding details based on the job requisition number provided in each production line list
P1_df = df[df['Requisition'].isin(P1_All)]
P2_df = df[df['Requisition'].isin(P2_All)]
P4_df = df[df['Requisition'].isin(P4_All)]

#Drop Index and Reset
P1_df = P1_df.reset_index(drop=True)
P2_df = P2_df.reset_index(drop=True)
P4_df = P4_df.reset_index(drop=True)

#Optimal Jobs to put on Each Line to Maximize Weight Tonnage
P1 = []
P2 = []
P4 = []

#Total Tonnage Capacities and Completion Times at each Line
P1_Tonnage_Capacity = []
P2_Tonnage_Capacity = []
P4_Tonnage_Capacity = []
P1_Completion_Time = []
P2_Completion_Time = []
P4_Completion_Time = []

#Tonnage Optimization Model (P1)

#Data Inputs:
#List of Job Requisitions
P1_Jobs = P1_df["Requisition"].values.tolist()

#Timeslots
timeslots = range(len(P1_Jobs))

#Weight Tonnage of each Job
W1 = P1_df["Tons"].values.tolist()

#Estimated Time of a Job
JT1 = P1_df["Estimated Time (Hours)"].values.tolist()

#Critical Date Hours Available of a Job
CD1 = P1_df["Critical Date Hours"].values.tolist()

#Job Priority of similar Tonnaged Jobs
Priority1 = P1_df["Priority"].values.tolist()
Priority1 = np.divide(Priority1, 1000000)

#--------------------------------------------------------------------------------------------
#Optimization Model: Selecting the jobs to run on each line that would maximize line tonnage
#Install and import packages:
from mip import *
import math

#Model:
m = Model(name='Maximize Tonnage Model (P1)', sense=MAXIMIZE)

#Variables:
X1 = [[m.add_var(name = "X_({},{})".format(j,t) , var_type = BINARY) for t in range(len(timeslots))] for j in range(len(P1_Jobs))] #Decision Variable: whether a job j is assigned to a timeslot t
T1 = m.add_var(name = "Tonnage on P1")
TP1 = m.add_var(name = "Total Priority")
PRIO1 = [m.add_var(name = "Job Priority") for t in range(len(timeslots))]
L1 = [m.add_var(name = "Delivery Lead Time") for t in range(len(timeslots))]
WT1 = [m.add_var(name = "Weight Tonnage at a Timeslot") for t in range(len(timeslots))]
CT1 = [m.add_var(name = "Estimated Completion Time at a Timeslot") for t in range(len(timeslots))]
ET1 = [m.add_var(name = "Estimated Time at a Timeslot") for t in range(len(timeslots))]
CDH1 = [m.add_var(name = "Critical Date Hours Available") for t in range(len(timeslots))]

#Objective Function:
m.objective = T1 - TP1

#Constraints:
#Constraint 1: Determining the job weight at each timeslot
for t in range(len(timeslots)):
  m += WT1[t] == xsum(W1[j]*X1[j][t] for j in range(len(P1_Jobs)))

#Constraint 2: Tonnage (T1) equals to the job weight in each timeslot
m += T1 == xsum(WT1[t] for t in range(len(timeslots)))

#Constraint 3: Prioritizing earlier critical dates for jobs with similar tonnage weights
for t in range(len(timeslots)):
  m += PRIO1[t] == xsum(Priority1[j]*X1[j][t] for j in range(len(P1_Jobs)))

#Constraint 4: Total Priority value
m += TP1 == xsum(PRIO1[t] for t in range(len(timeslots)))

#Constraint 5: Total Tonnage Capacity must be less than or equal to maximum tonnage capacity (Optional: uncomment only if line tonnage capacity limit is a hard constraint)
#m += T1 <= total_ton_cap

#Constraint 6: Determining Estimated Time at each Timeslot
for t in range(len(timeslots)):
  m += ET1[t] == xsum(JT1[j]*X1[j][t] for j in range(len(P1_Jobs)))

#Constraint 7: Total Completion Time must be less than or equal to the Total Available Hours
m += xsum(ET1[t] for t in range(len(timeslots))) <= Total_Hours

#Constraint 8: Determining Completion Time at each Timeslot
for t in range(len(timeslots)):
  m += CT1[t] == sum(ET1[:t+1])

#Constraint 9: Determining the Critical Date Hours Available at each Timeslot
for t in range(len(timeslots)):
  m += CDH1[t] == xsum(CD1[j]*X1[j][t] for j in range(len(P1_Jobs)))

#Constraint 10: Determining Lead Time of each Job at each Timeslot
for t in range(len(timeslots)):
  m += L1[t] == CDH1[t] - CT1[t]

#Constraint 11: Ensuring lead time is greater than or equal to 0 at each timeslot
for t in range(len(timeslots)):
  m += L1[t] >= 0

#Constraint 12: Each job can only be assigned to one timeslot and each timeslot can only be assigned to one job.
for j in range(len(P1_Jobs)):
  m += xsum(X1[j][t] for t in range(len(timeslots))) <= 1

for t in range(len(timeslots)):
  m += xsum(X1[j][t] for j in range(len(P1_Jobs))) <= 1

#--------------------------------------------------------------------------------------------
#Solve Optimization Model: Maximize Tonnage Model (P1)
status = m.optimize()

if status == OptimizationStatus.OPTIMAL:
  print("Optimal Schedule: P1")
  print("Objective Value:", m.objective_value)
  for j in range(len(P1_Jobs)):
    for t in range(len(timeslots)):
      if X1[j][t].x == 1:
        print("Job Requisition: [{}] || Tonnage: {} (tons) Completion Time: {} (line hours) Delivery Lead Time: {} (hours) --> Decision: {}".format(P1_Jobs[j], round(WT1[t].x,2), round(CT1[t].x,2), round(L1[t].x,2), X1[j][t].x))
        P1.append(P1_Jobs[j])
        P1_Tonnage_Capacity.append(WT1[t].x)
        P1_Completion_Time.append(CT1[t].x)

elif status == OptimizationStatus.UNBOUNDED:
  print("Unbounded")

elif status == OptimizationStatus.INFEASIBLE:
  print("Infeasible")

else:
  print("ERROR, unexpected result")

#Tonnage Optimization Model (P2)

#Data Inputs:
#List of Job Requisitions
P2_Jobs = P2_df["Requisition"].values.tolist()

#Timeslots
timeslots = range(len(P2_Jobs))

#Weight Tonnage of each Job
W2 = P2_df["Tons"].values.tolist()

#Estimated Time of a Job
JT2 = P2_df["Estimated Time (Hours)"].values.tolist()

#Critical Date Hours Available of a Job
CD2 = P2_df["Critical Date Hours"].values.tolist()

#Job Priority of similar Tonnaged Jobs
Priority2 = P2_df["Priority"].values.tolist()
Priority2 = np.divide(Priority2, 1000000)

#--------------------------------------------------------------------------------------------
#Optimization Model: Selecting the jobs to run on each line that would maximize line tonnage
#Install and import packages:
from mip import *
import math

#Model:
m = Model(name='Maximize Tonnage Model (P2)', sense=MAXIMIZE)

#Variables:
X2 = [[m.add_var(name = "X_({},{})".format(j,t) , var_type = BINARY) for t in range(len(timeslots))] for j in range(len(P2_Jobs))]
T2 = m.add_var(name = "Tonnage on P2")
TP2 = m.add_var(name = "Total Priority")
PRIO2 = [m.add_var(name = "Job Priority") for t in range(len(timeslots))]
L2 = [m.add_var(name = "Delivery Lead Time") for t in range(len(timeslots))]
WT2 = [m.add_var(name = "Weight Tonnage at a Timeslot") for t in range(len(timeslots))]
CT2 = [m.add_var(name = "Estimated Completion Time at a Timeslot") for t in range(len(timeslots))]
ET2 = [m.add_var(name = "Estimated Time at a Timeslot") for t in range(len(timeslots))]
CDH2 = [m.add_var(name = "Critical Date Hours Available") for t in range(len(timeslots))]

#Objective Function:
m.objective = T2 - TP2

#Constraints:
#Constraint 1: Determining the job weight at each timeslot
for t in range(len(timeslots)):
  m += WT2[t] == xsum(W2[j]*X2[j][t] for j in range(len(P2_Jobs)))

#Constraint 2: Tonnage (T2) equals to the job weight in each timeslot
m += T2 == xsum(WT2[t] for t in range(len(timeslots)))

#Constraint 3: Prioritizing earlier critical dates for jobs with similar tonnage weights
for t in range(len(timeslots)):
  m += PRIO2[t] == xsum(Priority2[j]*X2[j][t] for j in range(len(P2_Jobs)))

#Constraint 4: Total Priority value
m += TP2 == xsum(PRIO2[t] for t in range(len(timeslots)))

#Constraint 5: Total Tonnage Capacity must be less than or equal to maximum tonnage capacity (Optional: uncomment only if line tonnage capacity limit is a hard constraint)
#m += T2 <= 60*24

#Constraint 6: Determining Estimated Time at each Timeslot
for t in range(len(timeslots)):
  m += ET2[t] == xsum(JT2[j]*X2[j][t] for j in range(len(P2_Jobs)))

#Constraint 7: Total Completion Time must be less than or equal to the Total Available Hours
m += xsum(ET2[t] for t in range(len(timeslots))) <= Total_Hours

#Constraint 8: Determining Completion Time at each Timeslot
for t in range(len(timeslots)):
  m += CT2[t] == sum(ET2[:t+1])

#Constraint 9: Determining the Critical Date Hours Available at each Timeslot
for t in range(len(timeslots)):
  m += CDH2[t] == xsum(CD2[j]*X2[j][t] for j in range(len(P2_Jobs)))

#Constraint 10: Determining Lead Time of each Job at each Timeslot
for t in range(len(timeslots)):
  m += L2[t] == CDH2[t] - CT2[t]

#Constraint 11: Ensuring lead time is greater than or equal to 0 at each timeslot
for t in range(len(timeslots)):
  m += L2[t] >= 0

#Constraint 12: Each job can only be assigned to one timeslot and each timeslot can only be assigned to one job.
for j in range(len(P2_Jobs)):
  m += xsum(X2[j][t] for t in range(len(timeslots))) <= 1

for t in range(len(timeslots)):
  m += xsum(X2[j][t] for j in range(len(P2_Jobs))) <= 1

#--------------------------------------------------------------------------------------------
#Solve Optimization Model: Maximize Tonnage Model (P2)
status = m.optimize()

if status == OptimizationStatus.OPTIMAL:
  print("Optimal Schedule: P2")
  print("Objective Value:", m.objective_value)
  for j in range(len(P2_Jobs)):
    for t in range(len(timeslots)):
      if X2[j][t].x == 1:
        print("Job Requisition: [{}] || Tonnage: {} (tons) Completion Time: {} (line hours) Delivery Lead Time: {} (hours) --> Decision: {}".format(P2_Jobs[j], round(WT2[t].x,2), round(CT2[t].x,2), round(L2[t].x,2), X2[j][t].x))
        P2.append(P2_Jobs[j])
        P2_Tonnage_Capacity.append(WT2[t].x)
        P2_Completion_Time.append(CT2[t].x)

elif status == OptimizationStatus.UNBOUNDED:
  print("Unbounded")

elif status == OptimizationStatus.INFEASIBLE:
  print("Infeasible")

else:
  print("ERROR, unexpected result")

#Tonnage Optimization Model (P4)

#Data Inputs:
#List of Job Requisitions
P4_Jobs = P4_df["Requisition"].values.tolist()

#Timeslots
timeslots = range(len(P4_Jobs))

#Weight Tonnage of each Job
W4 = P4_df["Tons"].values.tolist()

#Estimated Time of a Job
JT4 = P4_df["Estimated Time (Hours)"].values.tolist()

#Critical Date Hours Available of a Job
CD4 = P4_df["Critical Date Hours"].values.tolist()

#Job Priority of similar Tonnaged Jobs
Priority4 = P4_df["Priority"].values.tolist()
Priority4 = np.divide(Priority4, 1000000)

#--------------------------------------------------------------------------------------------
#Optimization Model: Selecting the jobs to run on each line that would maximize line tonnage
#Install and import packages:
from mip import *
import math

#Model:
m = Model(name='Maximize Tonnage Model (P4)', sense=MAXIMIZE)

#Variables:
X4 = [[m.add_var(name = "X_({},{})".format(j,t) , var_type = BINARY) for t in range(len(timeslots))] for j in range(len(P4_Jobs))]
T4 = m.add_var(name = "Tonnage on P4")
TP4 = m.add_var(name = "Total Priority")
PRIO4 = [m.add_var(name = "Job Priority") for t in range(len(timeslots))]
L4 = [m.add_var(name = "Delivery Lead Time") for t in range(len(timeslots))]
WT4 = [m.add_var(name = "Weight Tonnage at a Timeslot") for t in range(len(timeslots))]
CT4 = [m.add_var(name = "Estimated Completion Time at a Timeslot") for t in range(len(timeslots))]
ET4 = [m.add_var(name = "Estimated Time at a Timeslot") for t in range(len(timeslots))]
CDH4 = [m.add_var(name = "Critical Date Hours Available") for t in range(len(timeslots))]

#Objective Function:
m.objective = T4 - TP4

#Constraints:
#Constraint 1: Determining the job weight at each timeslot
for t in range(len(timeslots)):
  m += WT4[t] == xsum(W4[j]*X4[j][t] for j in range(len(P4_Jobs)))

#Constraint 2: Tonnage (T4) equals to the job weight in each timeslot
m += T4 == xsum(WT4[t] for t in range(len(timeslots)))

#Constraint 3: Prioritizing earlier critical dates for jobs with similar tonnage weights
for t in range(len(timeslots)):
  m += PRIO4[t] == xsum(Priority4[j]*X4[j][t] for j in range(len(P4_Jobs)))

#Constraint 4: Total Priority value
m += TP4 == xsum(PRIO4[t] for t in range(len(timeslots)))

#Constraint 5: Total Tonnage Capacity must be less than or equal to maximum tonnage capacity (Optional: uncomment only if line tonnage capacity limit is a hard constraint)
#m += T4 <= 60*24

#Constraint 6: Determining Estimated Time at each Timeslot
for t in range(len(timeslots)):
  m += ET4[t] == xsum(JT4[j]*X4[j][t] for j in range(len(P4_Jobs)))

#Constraint 7: Total Completion Time must be less than or equal to the Total Available Hours
m += xsum(ET4[t] for t in range(len(timeslots))) <= Total_Hours

#Constraint 8: Determining Completion Time at each Timeslot
for t in range(len(timeslots)):
  m += CT4[t] == sum(ET4[:t+1])

#Constraint 9: Determining the Critical Date Hours Available at each Timeslot
for t in range(len(timeslots)):
  m += CDH4[t] == xsum(CD4[j]*X4[j][t] for j in range(len(P4_Jobs)))

#Constraint 10: Determining Lead Time of each Job at each Timeslot
for t in range(len(timeslots)):
  m += L4[t] == CDH4[t] - CT4[t]

#Constraint 11: Ensuring lead time is greater than or equal to 0 at each timeslot
for t in range(len(timeslots)):
  m += L4[t] >= 0

#Constraint 12: Each job can only be assigned to one timeslot and each timeslot can only be assigned to one job
for j in range(len(P4_Jobs)):
  m += xsum(X4[j][t] for t in range(len(timeslots))) <= 1

for t in range(len(timeslots)):
  m += xsum(X4[j][t] for j in range(len(P4_Jobs))) <= 1

#--------------------------------------------------------------------------------------------
#Solve Optimization Model: Maximize Tonnage Model (P4)
status = m.optimize()

if status == OptimizationStatus.OPTIMAL:
  print("Optimal Schedule: P4")
  print("Objective Value:", m.objective_value)
  for j in range(len(P4_Jobs)):
    for t in range(len(timeslots)):
      if X4[j][t].x == 1:
        print("Job Requisition: [{}] || Tonnage: {} (tons) Completion Time: {} (line hours) Delivery Lead Time: {} (hours) --> Decision: {}".format(P4_Jobs[j], round(WT4[t].x,2), round(CT4[t].x,2), round(L4[t].x,2), X4[j][t].x))
        P4.append(P4_Jobs[j])
        P4_Tonnage_Capacity.append(WT4[t].x)
        P4_Completion_Time.append(CT4[t].x)

elif status == OptimizationStatus.UNBOUNDED:
  print("Unbounded")

elif status == OptimizationStatus.INFEASIBLE:
  print("Infeasible")

else:
  print("ERROR, unexpected result")

#Creating a dataframe for each production line, including only the job requisition and corresponding details based on the job requisition number provided in each production line list
P1_data = df[df['Requisition'].isin(P1)]
P2_data = df[df['Requisition'].isin(P2)]
P4_data = df[df['Requisition'].isin(P4)]

#Drop index and Reset
P1_data = P1_data.reset_index(drop=True)
P2_data = P2_data.reset_index(drop=True)
P4_data = P4_data.reset_index(drop=True)

#Optimal Jobs to sequence and schedule on each Line
P1_Optimal = []
P2_Optimal = []
P4_Optimal = []

#Job Sequencing Optimization Model (P1)

#Data Inputs:
#Job Requisitions
P1_Jobs = P1_data["Requisition"].values.tolist()

#Timeslots
timeslots = range(len(P1_Jobs))

#Constant value for Delivery Lead time significance
L1_significance = 0.001

#Depth values of each Job
d1 = P1_data["Max Depth (Inches)"].values.tolist()

#Estimated Time for each Job
JT1 = P1_data["Estimated Time (Hours)"].values.tolist()

#Critical Date Hours Available for each Job
CD1 = P1_data["Critical Date Hours"].values.tolist()

#--------------------------------------------------------------------------------------------
#Optimization Model: Job Seqeuencing to Minimize Changeover and Minimize Lateness (P1)

#Model:
m = Model(name='Job Sequencing Optimization Model (P1)', sense=MINIMIZE)

#Variables:
Z1 = [m.add_var(name = "Depth Difference") for t in range(len(timeslots))]
L1 = [m.add_var(name = "Delivery Lead Time") for t in range(len(timeslots))]
S1 = [[m.add_var(name = "S_({},{})".format(j,t) , var_type = BINARY) for t in range(len(timeslots))] for j in range(len(P1_Jobs))] #decision variable
CT1 = [m.add_var(name = "Estimated Completion Time at a Timeslot") for t in range(len(timeslots))]
ET1 = [m.add_var(name = "Estimated Time at a Timeslot") for t in range(len(timeslots))]
CDH1 = [m.add_var(name = "Critical Date Hours Available") for t in range(len(timeslots))]
DT1 = [m.add_var(name = "Depth Value at a Timeslot") for t in range(len(timeslots))]

#Objective Function:
m.objective = xsum(Z1[t] - (L1_significance*L1[t]) for t in range(len(timeslots)))

#Constraints:
#Constraint #1: Determining depth value at each timeslot
for t in range(len(timeslots)):
  m += DT1[t] == xsum(d1[j]*S1[j][t] for j in range(len(P1_Jobs)))

#Constraint #2: Enforcing jobs are sequenced based on depth in descending order
for t in range(len(timeslots)-1):
  m += Z1[t] >= DT1[t+1] - DT1[t]

#Constraint #3: Determining Estimated Time at each Timeslot
for t in range(len(timeslots)):
  m += ET1[t] == xsum(JT1[j]*S1[j][t] for j in range(len(P1_Jobs)))

#Constraint #4: Determining Completion Time at each Timeslot
for t in range(len(timeslots)):
  m += CT1[t] == sum(ET1[:t+1])

#Constraint #5: Determining the Critical Date Hours Available at each Timeslot
for t in range(len(timeslots)):
  m += CDH1[t] == xsum(CD1[j]*S1[j][t] for j in range(len(P1_Jobs)))

#Constraint #6: Determining Lead Time of each Job at each Timeslot
for t in range(len(timeslots)):
  m += L1[t] == CDH1[t] - CT1[t]

#Constraint #7: Enforcing that a job must be completed before its critical date
for t in range(len(timeslots)):
  m += L1[t] >= 0

#Constraint #8: Each job can only be assigned to one timeslot and each timeslot can only be assigned to one job
for j in range(len(P1_Jobs)):
  m += xsum(S1[j][t] for t in range(len(timeslots))) == 1

for t in range(len(timeslots)):
  m += xsum(S1[j][t] for j in range(len(P1_Jobs))) == 1

#--------------------------------------------------------------------------------------------
#Solve Optimization Model: Job Seqeuencing to Minimize Changeover and Minimize Lateness (P1)
status = m.optimize()

if status == OptimizationStatus.OPTIMAL:
  print("Optimal Schedule")
  print("Objective Value:", m.objective_value)
  for t in range(len(timeslots)):
    for j in range(len(P1_Jobs)):
      if S1[j][t].x == 1:
        print("Job Requisition: [{}] Depth: [{}] Order: [{}] || Completion Time: {} (line hours) Delivery Lead Time: {} (hours)".format(P1_Jobs[j], DT1[t].x, timeslots[t], round(CT1[t].x,2), round(L1[t].x,2), S1[j][t].x))
        P1_Optimal.append(P1_Jobs[j])

elif status == OptimizationStatus.UNBOUNDED:
  print("Unbounded")

elif status == OptimizationStatus.INFEASIBLE:
  print("Infeasible")

else:
  print("ERROR, unexpected result")

#Job Sequencing Optimization Model (P2)

#Data Inputs:
#Job Requisitions
P2_Jobs = P2_data["Requisition"].values.tolist()

#Timeslots
timeslots = range(len(P2_Jobs))

#Constant value for Delivery Lead time significance
L2_significance = 0.001

#Depth values of each Job
d2 = P2_data["Max Depth (Inches)"].values.tolist()

#Estimated Time for each Job
JT2 = P2_data["Estimated Time (Hours)"].values.tolist()

#Critical Date Hours Available for each Job
CD2 = P2_data["Critical Date Hours"].values.tolist()

#--------------------------------------------------------------------------------------------
#Optimization Model: Job Seqeuencing to Minimize Changeover and Minimize Lateness (P2)

#Model:
m = Model(name='Job Sequencing Optimization Model (P2)', sense=MINIMIZE)

#Variables:
Z2 = [m.add_var(name = "Depth Difference") for t in range(len(timeslots))]
L2 = [m.add_var(name = "Delivery Lead Time") for t in range(len(timeslots))]
S2 = [[m.add_var(name = "S_({},{})".format(j,t) , var_type = BINARY) for t in range(len(timeslots))] for j in range(len(P2_Jobs))] #decision variable
CT2 = [m.add_var(name = "Estimated Completion Time at a Timeslot") for t in range(len(timeslots))]
ET2 = [m.add_var(name = "Estimated Time at a Timeslot") for t in range(len(timeslots))]
CDH2 = [m.add_var(name = "Critical Date Hours Available") for t in range(len(timeslots))]
DT2 = [m.add_var(name = "Depth Value at a Timeslot") for t in range(len(timeslots))]

#Objective Function:
m.objective = xsum(Z2[t] - (L2_significance*L2[t]) for t in range(len(timeslots)))

#Constraints:
#Constraint #1: Determining depth value at each timeslot
for t in range(len(timeslots)):
  m += DT2[t] == xsum(d2[j]*S2[j][t] for j in range(len(P2_Jobs)))

#Constraint #2: Enforcing jobs are sequenced based on depth in descending order
for t in range(len(timeslots)-1):
  m += Z2[t] >= DT2[t+1] - DT2[t]

#Constraint #3: Determining Estimated Time at each Timeslot
for t in range(len(timeslots)):
  m += ET2[t] == xsum(JT2[j]*S2[j][t] for j in range(len(P2_Jobs)))

#Constraint #4: Determining Completion Time at each Timeslot
for t in range(len(timeslots)):
  m += CT2[t] == sum(ET2[:t+1])

#Constraint #5: Determining the Critical Date Hours Available at each Timeslot
for t in range(len(timeslots)):
  m += CDH2[t] == xsum(CD2[j]*S2[j][t] for j in range(len(P2_Jobs)))

#Constraint #6: Determining Lead Time of each Job at each Timeslot
for t in range(len(timeslots)):
  m += L2[t] == CDH2[t] - CT2[t]

#Constraint #7: Enforcing that a job must be completed before its critical date
for t in range(len(timeslots)):
  m += L2[t] >= 0

#Constraint #8: Each job can only be assigned to one timeslot and each timeslot can only be assigned to one job
for j in range(len(P2_Jobs)):
  m += xsum(S2[j][t] for t in range(len(timeslots))) == 1

for t in range(len(timeslots)):
  m += xsum(S2[j][t] for j in range(len(P2_Jobs))) == 1

#--------------------------------------------------------------------------------------------
#Solve Optimization Model: Job Seqeuencing to Minimize Changeover and Minimize Lateness (P2)
status = m.optimize()

if status == OptimizationStatus.OPTIMAL:
  print("Optimal Schedule")
  print("Objective Value:", m.objective_value)
  for t in range(len(timeslots)):
    for j in range(len(P2_Jobs)):
      if S2[j][t].x == 1:
        print("Job Requisition: [{}] Depth: [{}] Order: [{}] || Completion Time: {} (line hours) Delivery Lead Time: {} (hours)".format(P2_Jobs[j], DT2[t].x,timeslots[t], round(CT2[t].x,2), round(L2[t].x,2), S2[j][t].x))
        P2_Optimal.append(P2_Jobs[j])

elif status == OptimizationStatus.UNBOUNDED:
  print("Unbounded")

elif status == OptimizationStatus.INFEASIBLE:
  print("Infeasible")

else:
  print("ERROR, unexpected result")

#Job Sequencing Optimization Model (P4)

#Data Inputs:
#Job Requisitions
P4_Jobs = P4_data["Requisition"].values.tolist()

#Timeslots
timeslots = range(len(P4_Jobs))

#Constant value for Delivery Lead time significance
L4_significance = 0.001

#Depth values of each Job
d4 = P4_data["Max Depth (Inches)"].values.tolist()

#Estimated Time for each Job
JT4 = P4_data["Estimated Time (Hours)"].values.tolist()

#Critical Date Hours Available for each Job
CD4 = P4_data["Critical Date Hours"].values.tolist()

#--------------------------------------------------------------------------------------------
#Optimization Model: Job Seqeuencing to Minimize Changeover and Minimize Lateness (P4)

#Model:
m = Model(name='Job Sequencing Optimization Model', sense=MINIMIZE)

#Variables:
Z4 = [m.add_var(name = "Depth Difference") for t in range(len(timeslots))]
L4 = [m.add_var(name = "Delivery Lead Time") for t in range(len(timeslots))]
S4 = [[m.add_var(name = "S_({},{})".format(j,t) , var_type = BINARY) for t in range(len(timeslots))] for j in range(len(P4_Jobs))] #decision variable
CT4 = [m.add_var(name = "Estimated Completion Time at a Timeslot") for t in range(len(timeslots))]
ET4 = [m.add_var(name = "Estimated Time at a Timeslot") for t in range(len(timeslots))]
CDH4 = [m.add_var(name = "Critical Date Hours Available") for t in range(len(timeslots))]
DT4 = [m.add_var(name = "Depth Value at a Timeslot") for t in range(len(timeslots))]

#Objective Function:
m.objective = xsum(Z4[t] - (L4_significance*L4[t]) for t in range(len(timeslots)))

#Constraints:
#Constraint #1: Determining depth value at each timeslot
for t in range(len(timeslots)):
  m += DT4[t] == xsum(d4[j]*S4[j][t] for j in range(len(P4_Jobs)))

#Constraint #2: Enforcing jobs are sequenced based on minimizing absolute changeover
for t in range(len(timeslots)-1):
  m += Z4[t] >= DT4[t+1] - DT4[t]

#Constraint #3: Determining Estimated Time at each Timeslot
for t in range(len(timeslots)):
  m += ET4[t] == xsum(JT4[j]*S4[j][t] for j in range(len(P4_Jobs)))

#Constraint #4: Determining Completion Time at each Timeslot
for t in range(len(timeslots)):
  m += CT4[t] == sum(ET4[:t+1])

#Constraint #5: Determining the Critical Date Hours Available at each Timeslot
for t in range(len(timeslots)):
  m += CDH4[t] == xsum(CD4[j]*S4[j][t] for j in range(len(P4_Jobs)))

#Constraint #6: Determining Lead Time of each Job at each Timeslot
for t in range(len(timeslots)):
  m += L4[t] == CDH4[t] - CT4[t]

#Constraint #7: Enforcing that a job must be completed before its critical date
for t in range(len(timeslots)):
  m += L4[t] >= 0

#Constraint #8: Each job can only be assigned to one timeslot and each timeslot can only be assigned to one job
for j in range(len(P4_Jobs)):
  m += xsum(S4[j][t] for t in range(len(timeslots))) == 1

for t in range(len(timeslots)):
  m += xsum(S4[j][t] for j in range(len(P4_Jobs))) == 1

#--------------------------------------------------------------------------------------------
#Solve Optimization Model: Job Seqeuencing to Minimize Changeover and Minimize Lateness (P4)
status = m.optimize()

if status == OptimizationStatus.OPTIMAL:
  print("Optimal Schedule")
  print("Objective Value:", m.objective_value)
  for t in range(len(timeslots)):
    for j in range(len(P4_Jobs)):
      if S4[j][t].x == 1:
        print("Job Requisition: [{}] Depth: [{}] Order: [{}] || Completion Time: {} (line hours) Delivery Lead Time: {} (hours)".format(P4_Jobs[j], DT4[t].x, timeslots[t], round(CT4[t].x,2), round(L4[t].x,2), S4[j][t].x))
        P4_Optimal.append(P4_Jobs[j])

elif status == OptimizationStatus.UNBOUNDED:
  print("Unbounded")

elif status == OptimizationStatus.INFEASIBLE:
  print("Infeasible")

else:
  print("ERROR, unexpected result")

print("Jobs to be scheduled on P1 line: {} || The tonnage capacity is {} tons and the estimated completion time is {} hours.".format(P1_Optimal, round(sum(P1_Tonnage_Capacity),2), round(max(P1_Completion_Time, default=0),2)))
print("Jobs to be scheduled on P2 line: {} || The tonnage capacity is {} tons and the estimated completion time is {} hours.".format(P2_Optimal, round(sum(P2_Tonnage_Capacity),2), round(max(P2_Completion_Time, default=0),2)))
print("Jobs to be scheduled on P4 line: {} || The tonnage capacity is {} tons and the estimated completion time is {} hours.".format(P4_Optimal, round(sum(P4_Tonnage_Capacity),2), round(max(P4_Completion_Time, default=0),2)))

#Calculating Total Tonnage Production and Man Hours per Ton
Total_Tonnage = sum(P1_Tonnage_Capacity, P2_Tonnage_Capacity, P4_Tonnage_Capacity)
print("Total Tonnage Production: {}".format(round(Total_Tonnage,2)))
print("Man Hours per Ton: {}".format(round((Total_Hours*14*3)/Total_Tonnage),2))

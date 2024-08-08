#Initial PPbyLine Dataset Cleaning
import tkinter
from tkinter import filedialog
import pandas as pd
from datetime import timedelta, datetime
import numpy as np

#Set Scheduling Start Date!
start_date = '08/07/2023'

root = tkinter.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()

#Select PPbyLine Input File
input_file = filedialog.askopenfilename(parent=root, initialdir="C:/", title="Select Scheduling Excel Input File", filetypes=[("Excel files", ".xlsx .xls .xlsm")])
output_file = filedialog.askopenfilename(parent=root, initialdir="C:/", title="Select Scheduling Excel Output File", filetypes=[("Excel files", ".xlsx .xls .xlsm")])

#Imports Input Excel File as a Dataframe
df = pd.read_excel(input_file, sheet_name=0)

#Code for Organizing Dataset (removing unnecessary columns, changing order of columns, and changing column names)
df.drop(columns=["ER", "UR", "BM", "SA", "Description", "Engineers"], inplace=True)
df.rename(columns={"Req": "Requisition"}, inplace=True)
df.rename(columns={"Project No": "Project Number"}, inplace=True)
df.rename(columns={"Qty": "Quantity"}, inplace=True)
df.rename(columns={"Max. Length": "Max Length (Feet)"}, inplace=True)
df.rename(columns={"Est. Time": "Estimated Time (Hours)"}, inplace=True)
df["Max Depth (Inches)"] = np.nan
df.insert(1, 'Critical Date', df.pop('Critical Date'))
df.insert(7, 'Max Depth (Inches)', df.pop('Max Depth (Inches)'))

#Remove products (x-bridging, horizontal bridging, joist subs, and bolts) that are not included in the P1, P2, and P4 production line process
df = df[df['Product'].str.contains("1122|HB|1123|Bolts|8120", case=False) == False]

#Changes empty numerical records to 0
df[['Quantity','Tons', 'Max Length (Feet)', 'Estimated Time (Hours)']] = df[['Quantity','Tons', 'Max Length (Feet)', 'Estimated Time (Hours)']].fillna(0)

#Remove ' apostrophe on Max Length values
df['Max Length (Feet)'] = [l.replace("'","") if isinstance(l, str) else l for l in df['Max Length (Feet)']]
df['Max Length (Feet)'] = df['Max Length (Feet)'].astype('int')

#Convert Estimated Time (man hours) to Line hours
df["Estimated Time (Hours)"] = df["Estimated Time (Hours)"].div(14)

#Removes Timestamp on Critical Date and removes any job requisitions that have a critical date prior 
#to scheduling start date (don't need to remove jobs with critical date outside of two week time frame because scheduler wants to schedule as much as possible)
start_date = datetime.strptime(start_date, '%m/%d/%Y').date()
#two_weeks_from_today = start_date + timedelta(14)
df['Critical Date'] = pd.to_datetime(df['Critical Date'])
df = df[df['Critical Date'] >= start_date.strftime('%Y-%m-%d')]
#df = df[df['Critical Date'].between(start_date.strftime('%Y-%m-%d'), two_weeks_from_today.strftime('%Y-%m-%d'), inclusive = 'both')]
df['Critical Date'] = df['Critical Date'].dt.date
df = df.reset_index(drop=True)

#Outputs dataframe data to selected Excel Output File
df.to_excel(output_file, sheet_name='Unassigned jobs', index=False)


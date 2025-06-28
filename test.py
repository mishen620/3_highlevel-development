import pandas as pd

#current_state=input()
# Read the first sheet
ds_curl_flexion = pd.read_csv('generalized_curl_flexion.csv')
ds_curl_extension = pd.read_csv('generalized_curl_extension.csv')

time=ds_curl_extension.columns[0]
angle=ds_curl_extension.columns[1]
#print(time,angle)


print(ds_curl_flexion.iloc[2, 1])
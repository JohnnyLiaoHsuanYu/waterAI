import pandas as pd
import numpy as np
import datetime
import sys

inflow_path = 'D:/2310011_Liao/水資源/resevior/all_waterresource/reservoir_data_over_time_all_water_source_inflow.csv'
outflow_path = 'D:/2310011_Liao/水資源/resevior/all_waterresource/reservoir_data_over_time_all_water_source_outflow.csv'
rain_path = 'D:/2310011_Liao/水資源/resevior/all_waterresource/reservoir_data_over_time_all_water_source_rain.csv'
percentage_path = 'D:/2310011_Liao/水資源/resevior/all_waterresource/reservoir_data_over_time_all_water_source_percentage.csv'
df_inflow = pd.read_csv(inflow_path)
df_outflow = pd.read_csv(outflow_path)
df_rain = pd.read_csv(rain_path)
df_percentage = pd.read_csv(percentage_path)


date = df_inflow.keys()[366:] # from 20120101
resevior_name = np.array(df_inflow.iloc[:, 0])
resevior_num = np.arange(0, 112)


df_inflow = df_inflow.iloc[:, 366:]
df_outflow = df_outflow.iloc[:, 366:]
df_rain = df_rain.iloc[:, 347:]
df_percentage = df_percentage.iloc[:, 366:]

# sys.exit()

all_inflow = [] 
all_outflow = []
all_rain = []
all_percentage = []
all_name = []
all_name_num = []
all_date = []
all_year = []
for i in range(df_inflow.shape[0]):  
    name___num = resevior_num[i]
    name___ = resevior_name[i]
    for j in range(date.shape[0]):  
        all_date.append(datetime.datetime.strptime((date[j]), "%Y%m%d").timetuple().tm_yday)
        all_year.append(datetime.datetime.strptime((date[j]), "%Y%m%d").year)
        all_name_num.append(name___num)
        all_name.append(name___)
        all_inflow.append(df_inflow.loc[i, date[j]])
        all_outflow.append(df_outflow.loc[i, date[j]])
        all_percentage.append(df_percentage.loc[i, date[j]])
        all_rain.append(df_rain.iloc[i, j:j+30].sum(axis=0, skipna=True))

all_rain = np.where(np.array(all_rain)==0, np.nan, np.array(all_rain))

data = {'percentage': all_percentage,
        'reseviornum': all_name_num,
        'date': all_date,
        'year': all_year,
        'precip': all_rain,
        'inflow': all_inflow,
        'outflow': all_outflow,
        'reseviorname': all_name,
        }
df_data = pd.DataFrame(data)

new_df_data = df_data.dropna(axis=0, how='any')

new_df_data.to_csv('D:/2310011_Liao/水資源/訓練資料/resevior/predict_resevior_percentage_withname.csv', index=False, encoding='utf-8-sig')


# fig,ax = plt.subplots(figsize=(7,4))
# plt.plot(np.array(df_data['precip']))
# plt.show()









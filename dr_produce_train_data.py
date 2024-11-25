import numpy as np
import pandas as pd 
import os
import sys
import glob
import datetime
from dateutil.relativedelta import relativedelta

readfile_path = 'D:/2310011_Liao/水資源/水情資料/水情_ETL.csv'
water_warn = pd.read_csv(readfile_path)

# county_path = r"D:\2310011_Liao\水資源\TCCIP\降雨"
# county_name = [f.name for f in os.scandir(county_path) if f.is_dir()][1:]

resevior_rd_path = 'D:/2310011_Liao/水資源/resevior/all_waterresource/reservoir_data_over_time_all_water_source_percentage.csv'
df_resevior = pd.read_csv(resevior_rd_path)

all_data = pd.DataFrame()
for ww in range(water_warn.shape[0]):
    ymd_org = int(water_warn.iloc[ww, :3]['日期'])
    place = water_warn.iloc[ww, :3]['區域']
    warn_level = water_warn.iloc[ww, :3]['水情警戒等級']

    if place == 0:
        county = ['基隆市', '新北市/基隆新北']
        resevior = ['新山水庫']
    elif place == 1:
        county = ['臺北市', '新北市/台北新北']
        resevior = ['翡翠水庫']
    elif place == 2:
        county = ['新北市/板新']
        resevior = ['翡翠水庫', '石門水庫']
    elif place == 3:
        county = ['桃園市', '新北市/林口']
        resevior = ['石門水庫']
    elif place == 4:
        county = ['新竹市', '新竹縣']
        resevior = ['寶山水庫', '寶山第二水庫']
    elif place == 5:
        county = ['苗栗縣']
        resevior = ['永和山水庫', '明德水庫']
    elif place == 6:
        county = ['臺中市']
        resevior = ['鯉魚潭水庫', '德基水庫', '石岡壩']
    elif place == 7:
        county = ['彰化縣']
        resevior = ['石岡壩', '湖山水庫', '集集攔河堰', '日月潭水庫']
    elif place == 8:
        county = ['南投縣']
        resevior = ['集集攔河堰', '日月潭水庫']
    elif place == 9:
        county = ['雲林縣']
        resevior = ['湖山水庫', '集集攔河堰', '日月潭水庫']
    elif place == 10:
        county = ['嘉義市','嘉義縣']
        resevior = ['仁義潭水庫', '蘭潭水庫', '曾文水庫', '烏山頭水庫'] # 87%[仁義潭&蘭潭平均] & 13%[曾文&烏山頭平均]
    elif place == 11:
        county = ['臺南市']
        resevior = ['南化水庫', '曾文水庫', '烏山頭水庫'] # 60%[南化] 40%[曾文&烏山頭平均]
    elif place == 12:
        county = ['高雄市']
        resevior = ['阿公店水庫', '澄清湖水庫', '鳳山水庫'] 
    elif place == 13:
        county = ['屏東縣']
        resevior = ['牡丹水庫']
    # elif place == 14:
    #     county = ['澎湖縣']
    #     resevior = []

    # number to datetime
    ymd = datetime.datetime.strptime(str(ymd_org), "%Y%m%d")
    date_yy, date_mm, date_dd = ymd.year, ymd.month, ymd.day

    # the day of the year
    day_of_year = ymd.timetuple().tm_yday

    # last month date
    last_ymd_precip = ymd - relativedelta(months=1)
    last_date_yy_p, last_date_mm_p, last_date_dd_p = last_ymd_precip.year, last_ymd_precip.month, last_ymd_precip.day

    # last 10-day date
    last_ymd_resevior = ymd - relativedelta(days=10)
    last_date_yy_re, last_date_mm_re, last_date_dd_re = last_ymd_resevior.year, last_ymd_resevior.month, last_ymd_resevior.day
    
    # datetime to number
    ymd_new_precip = int(last_ymd_precip.strftime("%Y%m%d"))
    ymd_new_resevior= int(last_ymd_resevior.strftime("%Y%m%d"))

    region_data = pd.DataFrame()
    for cc in county:
        print(cc)
        rd_path = 'D:/2310011_Liao/水資源/TCCIP/降雨/' + cc
        if date_yy == last_date_yy_p:
            f_name = glob.glob(rd_path + '/*_%s.csv'%(date_yy))[0]
            df = pd.read_csv(f_name)
            # print(ff)
            print(df.shape)
            lon_lat = df.iloc[:, :2]
            need_rain = df.loc[:, str(ymd_new_precip):str(ymd_org)]
            need_rain = need_rain.mask(need_rain<0, np.nan)

        elif date_yy != last_date_yy_p:
            f_name_1 = glob.glob(rd_path + '/*_%s.csv'%(date_yy))[0]
            f_name_2 = glob.glob(rd_path + '/*_%s.csv'%(last_date_yy_p))[0]
            df1 = pd.read_csv(f_name_1)
            df2 = pd.read_csv(f_name_2)
            # print(ff)
            print(df1.shape)
            print(df2.shape)
            lon_lat = df1.iloc[:, :2]
            need_rain_1 = df1.loc[:, :str(ymd_org)]
            need_rain_2 = df2.loc[:, str(ymd_new_precip):]
            need_rain = pd.concat([need_rain_1, need_rain_2], axis=1)
            need_rain = need_rain.mask(need_rain<0, np.nan)

        monthly_mean_need_rain = need_rain.sum(axis=1, skipna=True).to_frame()

        new_data = pd.concat([lon_lat, monthly_mean_need_rain], axis=1)
        region_data = pd.concat([region_data, new_data], axis=0)

    region_data.rename(columns={region_data.columns[2]: 'precip'}, inplace=True)


    if place == 10: # ['嘉義市','嘉義縣']
        region_data_resevior = []
        for rr in resevior:
            print(rr)
            df_re = df_resevior.loc[df_resevior.iloc[:,0]==rr]
            need_resevior = df_re.loc[:, str(ymd_new_resevior):str(ymd_org)]
            need_resevior[need_resevior<0] = np.nan
            mean_resevior = need_resevior.mean(axis=1, skipna=True).to_frame().values[0,0]
            region_data_resevior.append(mean_resevior)
            aaa = np.array(region_data_resevior)
        region_resevior_percentage = (((aaa[0]+aaa[1])/2)*0.87 + ((aaa[2]+aaa[3])/2)*0.13)/100

    elif place == 11: # ['臺南市']
        region_data_resevior = []
        for rr in resevior:
            print(rr)
            df_re = df_resevior.loc[df_resevior.iloc[:,0]==rr]
            need_resevior = df_re.loc[:, str(ymd_new_resevior):str(ymd_org)]
            need_resevior[need_resevior<0] = np.nan
            mean_resevior = need_resevior.mean(axis=1, skipna=True).to_frame().values[0,0]
            region_data_resevior.append(mean_resevior)
            aaa = np.array(region_data_resevior)
        region_resevior_percentage = ((aaa[0])*0.6 + ((aaa[1]+aaa[2])/2)*0.4)/100
    else:
        region_data_resevior = []
        for rr in resevior:
            print(rr)
            df_re = df_resevior.loc[df_resevior.iloc[:,0]==rr]
            need_resevior = df_re.loc[:, str(ymd_new_resevior):str(ymd_org)]
            need_resevior[need_resevior<0] = np.nan
            mean_resevior = need_resevior.mean(axis=1, skipna=True).to_frame().values[0,0]
            region_data_resevior.append(mean_resevior)
        region_resevior_percentage = np.nanmean(np.array(region_data_resevior))/100

    region_data.insert(0, 'level', warn_level)
    region_data.insert(1, 'date', day_of_year)
    region_data.insert(2, 'year', date_yy)
    region_data.insert(3, 'region', place)
    region_data.insert(4, 'resevior', region_resevior_percentage)

    all_data = pd.concat([all_data, region_data], axis=0, ignore_index=True)


today = datetime.date.today()
formatted_today = today.strftime('%Y%m%d')

# save path
csv_file_path = 'D:/2310011_Liao/水資源/訓練資料/drought/training_data_%s.csv'%(formatted_today)

# save file
all_data.to_csv(csv_file_path, index=False)
import pickle
import pandas as pd
import numpy as np
import json as js
import os
import datetime
import sys
import time
import requests
import pdfplumber
import io
import re
import yaml
import ast
import concurrent.futures
from joblib import dump, load
from bs4 import BeautifulSoup
from datetime import timedelta
from dateutil.relativedelta import relativedelta


from lib.log_module import CustomLog
# from lib.get_resevior_data import get_dates_in_range, get_resevior_data



log_path = os.path.join("D:", "2310011_Liao", "水資源" , "done_model", "drought", "logg", "dr_model.log")
logger = CustomLog(log_path, "DROUGHT")


now = datetime.datetime.now()
time_noww = now.strftime('%Y/%m/%d %H:%M:%S')
print(' ')
print(time_noww)

################## basic setting ##################
def is_valid_date(dd):
    try:
        datetime.datetime.strptime(str(dd), "%Y%m%d")
        return True
    except ValueError:
        return False

count_name = ['基隆市、新北市(淡水區、三芝區、金山區、石門區、萬里區、貢寮區、瑞芳區、雙溪區、平溪區、深坑區、坪林區、石碇區、烏來區、汐止區的43里、新店區安康地區)',
              '臺北市、新北市(三重區、新店區、永和區、中和區及汐止區7個里)',
              '新北市板新地區(板橋區、新莊區、泰山區、五股區、蘆洲區、八里區、三峽區、鶯歌區、土城區、樹林區及三重區、中和區部份地區)', 
              '桃園市、新北市林口地區',
              '新竹縣市',
              '苗栗縣',
              '臺中市',
              '彰化縣',
              '南投縣',
              '雲林縣',
              '嘉義縣市',
              '臺南市',
              '高雄市',
              '屏東縣'
              ]


stations = [['基隆', '鞍部', '新北'],
            ['臺北', '竹子湖'],
            ['淡水', '新北'],
            ['新屋'], 
            ['新竹'],
            ['新竹', '臺中'],  
            ['臺中'], 
            ['田中'], 
            ['日月潭', '阿里山'],
            ['田中', '嘉義'],
            ['嘉義'],
            ['臺南'],
            ['高雄', '玉山'],
            ['高雄', '恆春'],
           ]


need_resevior_name = ['仁義潭水庫', '南化水庫', '寶山水庫', '寶山第二水庫', '德基水庫', '新山水庫', '日月潭水庫', '明德水庫', \
                      '曾文水庫', '永和山水庫', '湖山水庫', '澄清湖水庫', '烏山頭水庫', '牡丹水庫', '石岡壩', '石門水庫', '翡翠水庫', '蘭潭水庫', \
                      '阿公店水庫', '集集攔河堰', '鯉魚潭水庫', '鳳山水庫']

water_warn = ['水情正常', '水情稍緊', '一階限水', '二階限水', '三階限水']



# =================== input needed model & data ===================
class ModelDataLoader:
    def __init__(self, base_path, model_name):
        self.base_path = base_path
        self.model_reseviors = {}
        self.model_name = model_name

        self.load_region_data()
        self.load_resevior_data()
        self.load_resevior_model()
        self.load_main_model()

    def load_region_data(self):
        re_path = os.path.join(self.base_path, 'county/county_lonlat_drop_repeat_points_with_resevior.csv')
        self.region_df = pd.read_csv(re_path)
        self.test_region = list(self.region_df['region'])
        self.test_lon = list(self.region_df['LON'])
        self.test_lat = list(self.region_df['LAT'])
        self.test_resevior_name = list(self.region_df['resevior'].apply(ast.literal_eval))  # transform to list
        self.test_four_region = list(self.region_df['four_region'])

    def load_resevior_data(self):
        resevior_rd_ = os.path.join(self.base_path, '訓練資料/resevior/predict_resevior_percentage_withname.csv')
        self.resevior_df = pd.read_csv(resevior_rd_)
        self.resevior_nums = self.resevior_df.set_index('reseviorname')['reseviornum'].to_dict()

    def load_resevior_model(self):
        for renn in self.resevior_nums.keys():
            model_path = os.path.join(self.base_path, f'done_model/resevior/resevior_random_forest_20241104_{renn}.joblib')
            if os.path.exists(model_path):
                self.model_reseviors[renn] = load(model_path)

    def load_main_model(self):
        model_path = os.path.join(self.base_path, self.model_name)
        self.model_ = load(model_path)


path___ = 'D:/2310011_Liao/水資源/'
model_name_dr = 'done_model/drought/waterAI_XGBRegressor_new.joblib'
data_loader = ModelDataLoader(path___, model_name_dr)


# =================== input ymal data ===================
conf_path = 'D:/2310011_Liao/水資源/CODE/config/resevoir_config.yaml'
with open(conf_path, "r", encoding="utf-8") as file:
    resevior_config = yaml.safe_load(file)


# =================== precipitation data & future date ===================
TODAY = datetime.date.today()

last_Month = TODAY - relativedelta(months=1)
last_YY = last_Month.year
last_mm = last_Month.month

url_rain = f'https://www.cwa.gov.tw/V8/C/C/Statistics/MonthlyData/MOD/{last_YY}_{last_mm}.html?'
session = requests.Session()
response = session.get(url_rain)
CWA_record_rain = {}
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for i in soup.find_all('tr'):
        title = i.find('th', headers='subH-1').text
        rain_MMAvg = i.find('td', headers='H-2 subH-5').text
        try:
            CWA_record_rain[title] = float(rain_MMAvg)
        except:
            CWA_record_rain[title] = rain_MMAvg

# ============ tranform into the form we needed ============
need_precip_info = []
for iiii in range(3):
    county_grab = np.unique(np.array(data_loader.region_df.loc[data_loader.region_df['four_region']==iiii]['region']))
    region_data = []
    
    for cgi, cg in enumerate(county_grab):
        st_choose = stations[cg]
        st_c_data = []
        
        for stci , stc in enumerate(st_choose):
            if isinstance(CWA_record_rain[stc], float):
                    st_c_data.append(CWA_record_rain[stc])
        
        region_data.append(np.nanmean(np.array(st_c_data)))
        
    need_precip_info.append({'region_space': iiii,
                             'forcast_mean': np.nanmean(np.array(region_data))
                            })  


# ===========================
#url_resevior = 'https://fhy.wra.gov.tw/ReservoirPage_2011/StorageCapacity.aspx'
#last_Ten_days = TODAY - relativedelta(days=11)
#date_range = get_dates_in_range(int(last_Ten_days.strftime('%Y%m%d')), int((TODAY-relativedelta(days=1)).strftime('%Y%m%d')))
#resevior_rain = get_resevior_data(url_resevior, date_range)        


# ============= get the future date==============
url_time = 'https://www.cwa.gov.tw/Data/fcst_pdf/FW14.pdf'
response = requests.get(url_time)
if response.status_code == 200:
    # tranform to the text 
    pdf_file = io.BytesIO(response.content)
    date_report = []
    precip_data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            text = page.extract_text()
            precip_data.append(tables)
            date_report.append(text)
else:
    print(f"Error: Unable to fetch the PDF. Status code: {response.status_code}")

need_time_info = []
for datei, datej in enumerate(np.array(date_report)[0].split()):
    if '發布日期' in datej:
        # print(datej)
        need_time_info.append(datej)
    if '有效期間' in datej:
        # print(datej)
        need_time_info.append(datej)
print('%s'%(need_time_info[0]))
print('%s'%(need_time_info[1]))
print('%s'%(need_time_info[2]))


match_year = [int(my)+1911 for my in re.findall(r'(\d{1,3})年', need_time_info[1])]
match_date = re.findall(r'(\d{1,2})月(\d{1,2})日', need_time_info[1])
if match_date:
    d_111 = datetime.datetime(match_year[0], int(match_date[0][0]), int(match_date[0][1]))
    d_444 = datetime.datetime(match_year[0], int(match_date[1][0]), int(match_date[1][1]))
    
    middle_date = d_111 + (d_444 - d_111)/2
    
    d_222 = d_111 + (middle_date - d_111)/2
    d_333 = middle_date + (d_444 - middle_date)/2

date_time = [int(d_111.strftime("%Y%m%d")),
             int(d_222.strftime("%Y%m%d")),
             int(middle_date.strftime("%Y%m%d")),
             int(d_333.strftime("%Y%m%d")),
             int(d_444.strftime("%Y%m%d")) 
            ]


# =================== PRDICTION ===================
def process_region(inn, test_region, test_resevior_name, test_four_region, resevior_nums, model_reseviors, test_date, test_yy, need_precip_info):
    place_num = test_region[inn]
    resevior = test_resevior_name[inn]
    precip = need_precip_info[test_four_region[inn]]['forcast_mean']
    
    ##### average prediction of the resevior in the past ten days #####
    pred_resevior = []
    for re_nn in resevior:
        resevior_train_data = []
        for iii in range(9, -1, -1):
            resevior_num = resevior_nums[re_nn]
            resevior_train_data.append({'reseviornum': resevior_num,
                                        'date': test_date-iii,
                                        'year': test_yy,
                                        'precip': precip})
            df_resevior_train_data = pd.DataFrame(resevior_train_data)
            
        model_re = model_reseviors[re_nn]
        a_resevior_ten_days = model_re.predict(df_resevior_train_data)/100
            
        pred_resevior.append(np.nanmean(a_resevior_ten_days))
    
    if place_num == 10:
        pred_resevior_mean = ((pred_resevior[0]+pred_resevior[1])/2)*0.87 + ((pred_resevior[2]+pred_resevior[3])/2)*0.13
    elif place_num == 11:
        pred_resevior_mean = pred_resevior[0]*0.6 + ((pred_resevior[1]+pred_resevior[2])/2)*0.4
    else:
        pred_resevior_mean = np.nanmean(np.array(pred_resevior))

    return pred_resevior_mean, precip, resevior

# =========== LET'S START THE PREDICTION ===========
ppprediction = np.full([len(date_time), data_loader.region_df.shape[0]], np.nan)
test_resevior_all = np.full([len(date_time), data_loader.region_df.shape[0]], np.nan)
test_precip_all = np.full([len(date_time), data_loader.region_df.shape[0]], np.nan)

for dd, dateeee in enumerate(date_time):
    print(dateeee)
    test_date = datetime.datetime.strptime(str(dateeee), "%Y%m%d").timetuple().tm_yday
    test_yy = datetime.datetime.strptime(str(dateeee), "%Y%m%d").year


    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor () as executor:
        results = list(executor.map(lambda inn: process_region(inn, data_loader.test_region, data_loader.test_resevior_name, data_loader.test_four_region, \
                                                               data_loader.resevior_nums, data_loader.model_reseviors, test_date, test_yy, need_precip_info), \
                                                               range(data_loader.region_df.shape[0])))

    test_resevior, test_precip, test_resevior_name__ = zip(*results) 
    
    # input data
    data = {'date': test_date,
            'year': test_yy, 
            'region': data_loader.test_region,
            'resevior': test_resevior,
            'LON': data_loader.test_lon,
            'LAT': data_loader.test_lat,
            'precip': test_precip,
            }
    test_data = pd.DataFrame(data)
    end_time = time.time()
    print('COST TIME:', (end_time - start_time)/60, 'min')


    ################## load model ##################
    # prediction
    prediction = data_loader.model_.predict(test_data)
    print(prediction)
    
    ppprediction[dd, :] = prediction   
    test_resevior_all[dd,:] = test_resevior
    test_precip_all[dd,:] = test_precip
    

test_resevior_all = np.where(test_resevior_all<0, 0, test_resevior_all)
test_precip_all = np.where(test_precip_all<0, 0, test_precip_all)

test_resevior_ = np.nanmean(test_resevior_all, 0)
test_precip_ = np.nanmean(test_precip_all, 0)


ppprediction = np.round(np.nanmean(ppprediction, 0), 0)
predictions = np.where(ppprediction<0, 0, ppprediction)


print(' ')
print(predictions)
################## output result ##################
predic_path = 'D:/2310011_Liao/水資源/done_model/prediction_result/'
os.makedirs(predic_path, exist_ok=True)
today = datetime.date.today()
formatted_today = today.strftime('%Y%m%d')
predic_name = model_name_dr[:-7]

jsonfile = open(predic_path + 'waterAI_result___.json', mode='w')
output = []
for i in range(test_data.shape[0]):
    output.append({'date': '%s-%s'%(date_time[0], date_time[-1]),
                   'lon': test_data['LON'][i],
                   'lat': test_data['LAT'][i],
                   'level': int(predictions[i]),
                   'precip': test_precip_[i],
                   'resevior': np.round(test_resevior_[i], 4)*100,
                   'resevior_name': test_resevior_name__[i],
    })
    
js.dump(output, jsonfile, ensure_ascii=False)
jsonfile.close()


logger.make_log(f'\n\
                ===================================\n\
                {data_loader.model_.__class__.__name__}\n\
                Information of the rain prediction from CWA\n\
                {need_time_info[0]}\n\
                {need_time_info[1]}\n\
                {need_time_info[2]}\n\
                used date:\n\
                {date_time[0]}\n\
                {date_time[1]}\n\
                {date_time[2]}\n\
                {date_time[3]}\n\
                {date_time[4]}'
                )

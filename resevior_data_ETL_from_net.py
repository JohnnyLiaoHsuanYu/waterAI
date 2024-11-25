import requests
from bs4 import BeautifulSoup
import os
import csv
import sys
import numpy as np
from datetime import datetime
import time
import json as js

# 目標網站的URL
url = "https://fhy.wra.gov.tw/ReservoirPage_2011/StorageCapacity.aspx"

# Session object to maintain the session
session = requests.Session()

# Function to get the necessary hidden inputs (__VIEWSTATE, __EVENTVALIDATION, etc.)
def get_hidden_inputs(soup):
    hidden_inputs = {}
    for input_tag in soup.find_all("input", type="hidden"):
        hidden_inputs[input_tag["name"]] = input_tag["value"]
    return hidden_inputs


def is_valid_date(yyyy, mm, dd):
    try:
        datetime(yyyy, mm, dd)
        return True
    except ValueError:
        return False



# 設定
download_folder = "D:/2310011_Liao/水資源/resevior/all_waterresource/"
os.makedirs(download_folder, exist_ok=True)
resevior_kind = '水庫及攔河堰' # 防汛重點水庫, 所有水庫, 水庫及攔河堰
para = '蓄水量' #蓄水量
output_data_name = 'reservoir_data_over_time_all_water_source_percentage'

years = range(2011, 2023)
months = range(1, 13)
days = range(1, 32)

if para == '水位':
    paramerter_num = 8
elif para == '蓄水量':
    paramerter_num = 10
elif para == '集水區降雨':
    paramerter_num = 3
elif para == '進水量':
    paramerter_num = 4
elif para == '出水量':
    paramerter_num = 5


print('執行' + resevior_kind + para)

reservoir_data = {}
# 迴圈選擇年、月、日
for year in years:
    for month in months:
        for day in days:
            # 取得初始頁面和隱藏欄位
            response = session.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # 獲取隱藏欄位
                hidden_inputs = get_hidden_inputs(soup)
                if not hidden_inputs: # 如果 hidden_inputs 是空值，再跑一次
                    print(f'{para} hidden_inputs 是空值，再跑一次')
                    response = session.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    time.sleep(2)
                    hidden_inputs = get_hidden_inputs(soup)
                
            
            if is_valid_date(year, month, day) == True:

                # 發送查詢請求
                query_payload = {
                    '__EVENTTARGET': 'ctl00$cphMain$btnQuery',
                    '__EVENTARGUMENT': '',
                    '__VIEWSTATE': hidden_inputs['__VIEWSTATE'],
                    '__VIEWSTATEGENERATOR': hidden_inputs['__VIEWSTATEGENERATOR'],
                    # '__EVENTVALIDATION': hidden_inputs['__EVENTVALIDATION'],
                    'ctl00$cphMain$cboSearch': resevior_kind,
                    'ctl00$cphMain$ucDate$cboYear': str(year),
                    'ctl00$cphMain$ucDate$cboMonth': str(month),
                    'ctl00$cphMain$ucDate$cboDay': str(day),
                }
                query_response = session.post(url, data=query_payload)

                # 檢查是否查詢成功
                if query_response.status_code == 200:
                    

                    # 更新隱藏欄位
                    soup = BeautifulSoup(query_response.content, 'html.parser')
                    hidden_inputs = get_hidden_inputs(soup)
                    if not hidden_inputs: # 如果 hidden_inputs 是空值，再跑一次
                        print(f'查詢的 {para} hidden_inputs 是空值，再跑一次')
                        soup = BeautifulSoup(query_response.content, 'html.parser')
                        time.sleep(2.5)
                        hidden_inputs = get_hidden_inputs(soup)


                    table = soup.find('table', {'id': 'ctl00_cphMain_gvList'})
                    
                    #ymd = str(year)+str(month)+str(day)

                    if table.find_all('tr')[2].find_all('td')[2].get_text(strip=True)[2:12] != '--迄:--':
                        date_time = table.find_all('tr')[2].find_all('td')[2].get_text(strip=True)[2:12]
                    elif table.find_all('tr')[3].find_all('td')[2].get_text(strip=True)[2:12] != '--迄:--':
                        date_time = table.find_all('tr')[3].find_all('td')[2].get_text(strip=True)[2:12]
                    elif table.find_all('tr')[4].find_all('td')[2].get_text(strip=True)[2:12] != '--迄:--':
                        date_time = table.find_all('tr')[4].find_all('td')[2].get_text(strip=True)[2:12]
                    elif table.find_all('tr')[5].find_all('td')[2].get_text(strip=True)[2:12] != '--迄:--':
                        date_time = table.find_all('tr')[5].find_all('td')[2].get_text(strip=True)[2:12]
                    else:
                        date_time = str(year) + '-' + str(month) + '-' + str(day)
                    
                    
                    print(' ')
                    print(date_time)
                    print(f"{para} Query successful for {date_time}")

                    for row in table.find_all('tr'):
                        cells = row.find_all('td')
                        #print(len(cells))
                        if len(cells) >= 11:  # 確保行中有足夠的數據單元格
                            # 提取水庫名稱（第一個<td>）和百分比（第十一個<td>）
                            reservoir_name = cells[0].get_text(strip=True)
                            parameter = cells[paramerter_num].get_text(strip=True)
                            parameter = parameter.replace(',', '')
                            # # 將提取的數據加入列表
                            # reservoir_data.append([date_time, reservoir_name, parameter])
                            if reservoir_name not in reservoir_data:
                                reservoir_data[reservoir_name] = {}

                            # 將日期和百分比數據記錄在該水庫的字典中
                            reservoir_data[reservoir_name][date_time] = parameter
                    
                else:
                    print(f"{para} Query failed for {year}-{month:02d}-{day:02d}")
                    print(' ')
# sys.exit()



################ 生成 json file ################
with open(download_folder + output_data_name + '.json', 'w', encoding='utf-8') as json_file:
    js.dump(reservoir_data, json_file, ensure_ascii=False, indent=4)

print(f"Json 文件已成功生成：{download_folder + output_data_name + '.json'}")

################ 生成 csv file ################
output_file_name = download_folder + output_data_name + '.csv'
with open(output_file_name, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)

    # 獲取所有日期標頭, set() 去除重複直
    date_headers_original = sorted(set(date for ddd in reservoir_data.values() for date in ddd.keys()))
    date_headers = []
    for idd in date_headers_original:
        aaaa = datetime.strptime(idd, "%Y-%m-%d") # string to datetime
        bbb = int(aaaa.strftime("%Y%m%d")) #datetime to number
        date_headers.append(bbb)

    # 寫入表頭
    writer.writerow([''] + date_headers)

    # 寫入每個水庫的名稱和其在不同日期的數據
    for reservoir_name, data in reservoir_data.items():
        row = [reservoir_name]
        for date in date_headers_original:
            if data.get(date, '') != '--':
                row.append(float(data.get(date, '')[:-1]))  # 如果當前日期無數據則留空
            else:
                row.append(np.nan)
        writer.writerow(row)

print(f"{para} CSV 文件已成功生成：{output_file_name}")


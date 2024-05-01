# activate mondodb atlas

import init
from datetime import datetime
import numpy as np
import pytz
import pandas as pd


eastern_tz = pytz.timezone('Europe/Moscow')

myclient = init.myclient

def mongodb_atlas(db,table_name):
    
    mydb = myclient[db]

    mycol = mydb[table_name]

    return mycol



def record_dialogue(db,record,name):
    print('begin of record')

    mydb = myclient[db]
    mycol = mydb[name]

    #mycol = mydb["customers"]
    now = datetime.now()
    #now_russia =   eastern_tz.localize(now)            

    mydict = { "name": record.name , "id": record.id, "message": record.message, "predicted":record.predicted, "response":record.response,"time": now ,'program':record.program}   
    
    mycol.insert_one(mydict)
    print('end of record')

def query(db,keylabel,collection):
    
    mydb = myclient[db]
    
    mycol = mydb[collection]
    
    myquery = { "tag": keylabel }
    
    mydoc = mycol.find(myquery)
    found = []
    
    for x in mydoc:
      found.append(x)
        
    return found
    


def show_program_name(db = 'itmo_data',access = False):
    mydb = myclient[db]
    list_db = mydb.list_collection_names()
    rm_list = []
    print(list_db)
    if (access == True):
        
        for name in list_db:
            num = list(mydb[name].find({'access':True}))           
            if(len(num) == 0):
                print(name)
                rm_list.append(name)

    list_db = list(set(list_db)-set(rm_list))
    time_order = []
    for name in list_db:
        time_order.append(list(mydb[name].find().sort({'time':-1}))[0]['time'])
    sorted(time_order)
    time_order_np = np.array(time_order)
    sort_index = np.argsort(time_order_np)
    new_arrage = []
    for i in sort_index:
        new_arrage.append(list_db[i])
    del list_db
    list_db = new_arrage
    
    return list_db

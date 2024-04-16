# activate mondodb atlas

import init
from datetime import datetime
import pytz


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

    mydict = { "name": record.name , "id": record.id, "message": record.message, "predicted":record.predicted, "response":record.response ,"time": now }   
    
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
    

def show_program_name(db = 'itmo_data'):
    mydb = myclient[db]
    list_db = mydb.list_collection_names()
    return list_db
   
# activate mondodb atlas

import init
from datetime import datetime
import pytz


eastern_tz = pytz.timezone('Europe/Moscow')

myclient = init.myclient

def mongodb_atlas(table_name):
    
    mydb = myclient["itmo_data"]

    mycol = mydb[table_name]

    return mycol



def record_dialogue(record,name):

    mydb = myclient["itmo_data"]
    mycol = mydb[name]

    #mycol = mydb["customers"]
    now = datetime.now()
    #now_russia =   eastern_tz.localize(now)            

    mydict = { "name": record.name , "id": record.id, "message": record.message, "predicted":record.predicted, "response":record.response ,"time": now }   
    
    x = mycol.insert_one(mydict)

def query(keylabel,collection):
    
    mydb = myclient["itmo_data"]
    
    mycol = mydb[collection]
    
    myquery = { "tag": keylabel }
    
    mydoc = mycol.find(myquery)
    found = []
    
    for x in mydoc:
      found.append(x)
        
    return found
    

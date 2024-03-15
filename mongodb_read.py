# activate mondodb atlas
from bson import ObjectId
import tokens
import main
import time
from datetime import datetime

myclient = tokens.myclient

def mongodb_atlas(table_name):
    
    mydb = myclient["itmo_data"]

    mycol = mydb[table_name]

    return mycol

# to force users to click button
def force_button_click(message):
    if not message.text.startswith('/'):
        main.bot.send_message(message.chat.id, 'You must click one of the buttons!')
        
    time.sleep(3)
    main.restart(message)


def record_dialogue(record,name):

    mydb = myclient["itmo_data"]
    mycol = mydb[name]
    #mycol = mydb["customers"]
    now = datetime.now()
    now_russia = tokens.eastern_tz.localize(now)            

    mydict = { "name": record.name , "id": record.id, "message": record.message, "predicted":record.predicted, "response":record.response ,"time": now_russia }   
    
    x = mycol.insert_one(mydict)

def query(keylabel):

    
    mydb = myclient["itmo_data"]
    
    mycol = mydb["original"]
    
    myquery = { "tag": keylabel }
    
    mydoc = mycol.find(myquery)
    found = []
    
    for x in mydoc:
      found.append(x)
        
    return found
    

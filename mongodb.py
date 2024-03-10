# activate mondodb atlas
from bson import ObjectId
from main import *
import main
import tokens
from datetime import datetime
from function import *

bot = tokens.bot
myclient = tokens.myclient

def mongodb_atlas(table_name):
    
    mydb = myclient["itmo_data"]

    mycol = mydb[table_name]

    return mycol

# to force users to click button
def force_button_click(message):
    if not message.text.startswith('/'):
        bot.send_message(message.chat.id, 'You must click one of the buttons!')
        
    time.sleep(3)
    restart(message)


# to update unknown new datasets
@bot.message_handler(commands=['itmoxxx'])
def update_unknown_datasets(message):
    mycol = mongodb_atlas('unknown_response') 
    unknows = []
    for x in mycol.find():
        unknows.append(x)   
    
    if(len(unknows)==0):
        bot.send_message(message.chat.id, 'No more Unknown datasets, redirect to menu')
        time.sleep(3)
        restart(message)
    else:
        bot.send_message(message.chat.id, f'{len(unknows)} Unknown datasets exist')
        max_length = 64 
         
        one_unknown = mycol.find_one()
        msg = one_unknown['message']
        
        indexs,_ = main.model_process(main.model,tokenizer,msg,max_length)
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        for i in range(len(indexs)):
            try:
                found = query(indexs[i])
                print(found[0]['responses'])
                markup.add(found[0]['responses'][0]) 
            except:
                pass
        markup.add('New Topic')
        mess = 'Please choose more correspondent topic'
        msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
        #send rest of unknown to next func
        bot.register_next_step_handler(msg, insert_delete_unknow,one_unknown)

def insert_delete_unknow(message,one_unknown):
    msg = message.text
    if(msg == 'New Topic'):
        mess = 'please think a response for question'
        msg = bot.send_message(message.chat.id,mess)
        bot.register_next_step_handler(msg, insert_new_topic,one_unknown)
    else:
        mycol = mongodb_atlas('unknown_response')
        mycol.delete_one({"_id":one_unknown['_id']}) 
        
        one_unknown.pop('_id')
        mycol = mongodb_atlas('new_response')
        mycol.insert_one(one_unknown) 
        time.sleep(3)
        update_unknown_datasets(message)


        
def insert_new_topic(message,one_unknown):
    msg = message.text

    mycol = mongodb_atlas('unknown_response')
    mycol.delete_one({"_id":one_unknown['_id']}) 

    one_unknown.pop('_id')
    one_unknown['response'] = msg
    mycol = mongodb_atlas('new_response')
    mycol.insert_one(one_unknown) 
        
    mycol = mongodb_atlas('original')
    label_num = None
    for i in mycol.find():
        label_num = i['tag']
        print(label_num)
    
    dict_example = {'tag':label_num,'patterns':one_unknown['message'],'responses':msg}
    
    mycol.insert_one(dict_example) 

    mess = 'successfully updated'
    bot.send_message(message.chat.id,mess)
    update_unknown_datasets(message)
    

def record_dialogue(record,name):

    mydb = myclient["itmo_data"]
    mycol = mydb[name]
    #mycol = mydb["customers"]
    now = datetime.now()
    now_russia = main.eastern_tz.localize(now)            

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
    

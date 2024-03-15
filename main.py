import time
from telebot import types
import telebot
import os
from torch import nn
from tokens import token , password

import mongodb_read 
import statistics 
import NLPmodel 
import function 

## ----- import pre-trained model

import torch

bot = telebot.TeleBot(token, parse_mode='None')

@bot.message_handler(commands=['admin'])  # Ответ на команду /start
def admin(message):
    mess = f'Password: '
    msg = bot.send_message(message.chat.id, mess)
    bot.register_next_step_handler(msg, admin_confirm)

class Record():
    def __init__(self):
        """
        :param args:
        """
        super(Record, self).__init__()
        self.name = None
        self.id = None
        self.message = None
        self.predicted = None
        self.response = None
        self.time = None

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        

class Args():
    embedding_dim = 1024
    hidden=512 
    num_labels = 23
    dropout=0.1
    
args = Args()
PATH = r"itmo_model.pt"
model = NLPmodel.QModel_Classifier(args.embedding_dim,args.hidden,args.num_labels,args.dropout)
tokenizer = NLPmodel.RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)

record = Record()

bot_name = 'ITMO BOT'

model.load_state_dict(torch.load(PATH))
model.eval()


def menu():
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton('help', callback_data='help'))   
        markup.add(types.InlineKeyboardButton('contact', callback_data='contact'))   
        markup.add(types.InlineKeyboardButton('main questions', callback_data='main questions'))  
        markup.add(types.InlineKeyboardButton('more questions', callback_data='more questions')) 
        markup.add(types.InlineKeyboardButton('application', callback_data='application'))    
        return markup



@bot.message_handler(commands=['start'])  # Ответ на команду /start
def start(message):
    mess = f'hi, <b>{message.from_user.first_name}</b>!\nI am - <b>{bot_name}</b>'

    #record name and id 
    record.id = message.chat.id
    record.name = message.from_user.first_name
    
    markup = menu()
    msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
    #bot.register_next_step_handler(msg, force_button_click)

def restart(message):
    mess = f'hi, <b>{message.from_user.first_name}</b>!\nI am - <b>{bot_name}</b>'
    markup = menu()
    msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
    #bot.register_next_step_handler(msg, force_button_click)

def model_process(model,tokenizer,sent,max_length):
    encoding = tokenizer(sent, return_tensors='pt', max_length=max_length, truncation=True)
    b_input_ids = encoding['input_ids']
    #token_type_ids = encoding['token_type_ids']
    attention_mask = encoding['attention_mask']
    outputs = model(b_input_ids, 
                    attention_mask=attention_mask)
    m = nn.Softmax(dim=1)
    outputs= m(outputs[0])
    output_list = outputs[0].detach().numpy()
    indexs = sorted(range(len(output_list)), key=lambda k: output_list[k], reverse=True)     
    probs = output_list[indexs]

    return indexs,probs

def more_questions(message):
    sent = message.text
    sent = sent.lower()
    
    sent = function.multiple_question_detect(sent)

    sent = sent.split('and')

    model.eval()
    max_length = 64
    print('item num :')
    print(len(sent))
    print()
    print(sent)
    
    function.model_decode(model,tokenizer,sent,max_length,message)

@bot.callback_query_handler(func=lambda call: True)
def message_reply(call):
    if call.data == 'contact':
        mess = 'click on mail to get contact with staff'
        bot.send_message(call.message.chat.id,mess)
        mess = """
Program coordinator --  aakarabintseva@itmo.ru
International office -- international@itmo.ru
Student office -- aakiseleva@itmo.ru
Migration office -- aakhalilova@itmo.ru 
"""
        bot.send_message(call.message.chat.id,mess)
        time.sleep(5)
        restart(call.message)
    elif call.data == 'help': 
        mess = """
contact --  to find Email address of specific staff in ITMO
main questions -- to answer most frequent questions from candidates
more questions -- to answer other questions
application -- to redirect to page to fill application
"""
        bot.send_message(call.message.chat.id,mess)
        time.sleep(5)
        restart(call.message)
    elif call.data == 'main questions':
        function.main_questions_function(call)
    elif call.data == 'more questions': 
        mess = f'please <b>clearly</b> write your questions \nIf multiple questions, please separate by . or and'
        msg = bot.send_message(call.message.chat.id, mess)
        bot.register_next_step_handler(msg, more_questions, parse_mode='html')
    elif call.data == 'application':
        linked_user = 'https://signup.itmo.ru/master'
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton(text='redirect to ITMO',
                            url=linked_user))
        mess = 'click to redirect to application form'
        bot.send_message(call.message.chat.id,mess, reply_markup=markup)
        time.sleep(5)
        restart(call.message)

    # admin mode
    elif call.data == 'correct_unknown_questions':
        mycol = mongodb_read.mongodb_atlas('unknown_response') 
        unknows = []
        for x in mycol.find():
            unknows.append(x)   
        
        if(len(unknows)==0):
            bot.send_message(call.message.chat.id, 'No  Unknown datasets, redirect to menu')
            time.sleep(3)
            restart(call.message)
        else:
            bot.send_message(call.message.chat.id, f'{len(unknows)} Unknown datasets exist')
            max_length = 64 
            
            ###
            one_unknown = mycol.find_one()
            print(one_unknown)

            try:
                while(one_unknown['message'] == None):
                    mycol.delete_one({"_id":one_unknown['_id']}) 
                    one_unknown = mycol.find_one()
            except:
                bot.send_message(call.message.chat.id, 'No  Unknown datasets, redirect to menu')
                time.sleep(3)
                restart(call.message)

            if(one_unknown != None):
                msg = one_unknown['message']
                
                indexs,_ = model_process(model,tokenizer,msg,max_length)
                markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
                for i in range(len(indexs)):
                    try:
                        found = mongodb_read.query(indexs[i])
                        markup.add(found[0]['responses'][0]) 
                        markup.add('New Topic')
                        markup.add('Go to Trash')
                        markup.add('None')
                    except:
                        pass
            
                mess = f'Please choose more correspondent topic to question : \n<b>{msg}</b>' 
                msg = bot.send_message(call.message.chat.id, mess, reply_markup=markup, parse_mode='html')
                #send rest of unknown to next func
                bot.register_next_step_handler(msg, insert_delete_unknow,one_unknown)

    elif call.data == 'statistics_plot':
        statistics.show_statistics(call.message)

        time.sleep(10)
        start(call.message)


# admin section 
        

def admin_confirm(message):
    if message.text == password :
        admin_start(message)
    else :
        mess = 'You dont have authentication , returning to home page'
        bot.send_message(message.chat.id, mess)
        time.sleep(3)
        restart(message)

def admin_menu():
        markup = types.InlineKeyboardMarkup() 
        markup.add(types.InlineKeyboardButton('correct_unknown_questions', callback_data='correct_unknown_questions'))    
        markup.add(types.InlineKeyboardButton('statistics', callback_data='statistics_plot'))    
        return markup


def admin_start(message):
    mess = f'hi, <b>{message.from_user.first_name}</b>!\nWelcome to admin mode'
    #record name and id   
    markup = admin_menu()
    msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
    #bot.register_next_step_handler(msg, force_button_click)




def update_unknown_datasets(message):

    mycol = mongodb_read.mongodb_atlas('unknown_response') 
    unknows = []
    for x in mycol.find():
        unknows.append(x)   
    
    if(len(unknows)==0):
        bot.send_message(message.chat.id, 'No  Unknown datasets, redirect to menu')
        time.sleep(3)
        start(message)
    else:
        bot.send_message(message.chat.id, f'{len(unknows)} Unknown datasets exist')
        max_length = 64 
         
        one_unknown = mycol.find_one()
        msg = one_unknown['message']
        
        indexs,_ = model_process(model,tokenizer,msg,max_length)
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        for i in range(len(indexs)):
            try:
                found = mongodb_read.query(indexs[i])
                print(found[0]['responses'])
                markup.add(found[0]['responses'][0]) 
                markup.add('New Topic')
                markup.add('None')
            except:
                pass

        mess = 'Please choose more correspondent response'
        msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
        #send rest of unknown to next func
        bot.register_next_step_handler(msg, insert_delete_unknow,one_unknown)

def insert_delete_unknow(message,one_unknown):
    msg = message.text
    try:
        if(msg == 'New Topic'):
            mess = 'please think a response for question'
            msg = bot.send_message(message.chat.id,mess)
            bot.register_next_step_handler(msg, insert_new_topic,one_unknown)
        elif(msg == 'None'):
            mess = 'back to menu'
            bot.send_message(message.chat.id,mess)
            time.sleep(3)
            admin_start(message)
        elif(msg == 'Go to Trash'):
            mycol = mongodb_read.mongodb_atlas('unknown_response')
            mycol.delete_one({"_id":one_unknown['_id']}) 

            mycol = mongodb_read.mongodb_atlas('trash')
            one_unknown.pop('_id')
            mycol.insert_one(one_unknown) 
            time.sleep(3)
            update_unknown_datasets(message)

        elif(msg == 'New Topic'):
            mycol = mongodb_read.mongodb_atlas('unknown_response')
            mycol.delete_one({"_id":one_unknown['_id']}) 
            
            one_unknown.pop('_id')
            mycol = mongodb_read.mongodb_atlas('new_response')
            mycol.insert_one(one_unknown) 
            time.sleep(3)
            update_unknown_datasets(message)
    except:
        mess = 'something not right, back to menu'
        bot.send_message(message.chat.id,mess)
        time.sleep(3)
        admin_start(message)

def insert_new_topic(message,one_unknown):
    msg = message.text

    mycol = mongodb_read.mongodb_atlas('unknown_response')
    mycol.delete_one({"_id":one_unknown['_id']}) 

    one_unknown.pop('_id')

    mycol = mongodb_read.mongodb_atlas('training_data')
    label_num = None
    for i in mycol.find().sort(''):
        label_num = i['tag']
        print(label_num)
    
    dict_example = {'tag':label_num,'patterns':one_unknown['message'],'responses':msg}
    mycol.insert_one(dict_example) 


    one_unknown['response'] = label_num
    mycol = mongodb_read.mongodb_atlas('new_response')
    mycol.insert_one(one_unknown) 
        

    mess = 'successfully updated'
    bot.send_message(message.chat.id,mess)
    update_unknown_datasets(message)

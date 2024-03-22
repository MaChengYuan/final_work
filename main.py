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
import pandas as pd
import random
import torch


class admin_mode_or_no():
    def __init__(self):
        """
        :param args:
        """
        super(admin_mode_or_no, self).__init__()
        self.in_admin = False

in_admin = admin_mode_or_no()

bot = telebot.TeleBot(token, parse_mode='None')

@bot.message_handler(commands=['admin'])  
def admin(message):
    delete_menu(message)
    print(in_admin.in_admin)
    if(in_admin.in_admin == True):
        mess = 'You are already in admin model'
        bot.send_message(message.chat.id, mess)
        time.sleep(3)
        admin_start(message)
    else:
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

file_exist = os.path.isfile(PATH)

model = NLPmodel.QModel_Classifier(args.embedding_dim,args.hidden,args.num_labels,args.dropout)
tokenizer = NLPmodel.RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)

record = Record()
bot_name = 'ITMO BOT'

msg_id = None

if(file_exist):
    model.load_state_dict(torch.load(PATH))
    model.eval()
else:
    print('Maintanance')

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
    # admin model false
    in_admin.in_admin = False

    mess = f'hi, <b>{message.from_user.first_name}</b>!\nI am - <b>{bot_name}</b>'
    mess += f'\n'

    #detect if there is response from staff    
    mycol = mongodb_read.mongodb_atlas('response_to_user')
    x = mycol.find({'id':message.chat.id}) 
    df = pd.DataFrame(list(x))
    if(len(df)==0):
        print(f'{message.chat.id} does not have response')
    else:
        list_of_res = df.response.tolist()
        list_of_ques = df.message.tolist()
        mess = f'you have responses from ITMO University'
        for i in range(len(list_of_res)):
            mycol.delete_one({'response':list_of_res[i]})
            mess += f'\n'
            mess += f'To your question : {list_of_ques[i]}'
            mess += f'\n'
            mess += f'\n'
            found = mongodb_read.query(list_of_res[i],'original')
            mess += random.choice(found)['responses'][0]
        
    #record name and id 
    record.id = message.chat.id
    record.name = message.from_user.first_name
    
    markup = menu()

    global msg_id
    msg_id = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html').message_id

    #bot.register_next_step_handler(msg, force_button_click)

def restart(message):
    global msg_id
    # admin model false
    in_admin.in_admin = False
    
    mess = f'hi, <b>{message.from_user.first_name}</b>!\nI am - <b>{bot_name}</b>'
    markup = menu()
    msg_id = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html').message_id
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
    
    sent = function.sep_by_and(sent)

    max_length = 64
    print('item num :')
    print(len(sent))
    print()
    print(sent)
    
    function.model_decode(sent,max_length,message)

def delete_menu(call):
    global msg_id
    try:
        bot.delete_message(call.from_user.id, msg_id)
        msg_id = None
    except:
        bot.delete_message(call.chat.id, msg_id)
        msg_id = None

@bot.callback_query_handler(func=lambda call: True)
def message_reply(call):
    if call.data == 'contact':
        delete_menu(call)
        mess = 'click on mail to get contact with staff'
        bot.send_message(call.message.chat.id,mess)

        # to delete last message
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
        delete_menu(call)
        mess = """
contact --  to find Email address of specific staff in ITMO
main questions -- to answer most frequent questions from candidates
more questions -- to answer other questions
application -- to redirect to page to fill application
"""
        bot.send_message(call.message.chat.id,mess)
        # to delete last message

        time.sleep(5)
        restart(call.message)
    elif call.data == 'main questions':
        delete_menu(call)
        function.main_questions_function(call)
    elif call.data == 'more questions': 
        delete_menu(call)
        if(file_exist):          
             
            mess = 'please <b>clearly</b> write your questions \nIf multiple questions, please separate by <b>. or and</b>'
            msg = bot.send_message(call.message.chat.id, mess,parse_mode='HTML')
            bot.register_next_step_handler(msg, more_questions)
        else:
            mess = f'The system is under maintenance, sorry for inconvenience'
            bot.send_message(call.message.chat.id, mess)
            time.sleep(5)
            restart(call.message)

    elif call.data == 'application':
        delete_menu(call)
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
        delete_menu(call)
        mycol = mongodb_read.mongodb_atlas('unknown_response') 
        unknows = []

        x = mycol.find() 
        df = pd.DataFrame(list(x))
        delete_df = df[df.message.isna()]
        
        for i in delete_df._id.tolist():
            mycol.delete_one({"_id":i}) 
        
        df = df.dropna(subset=['message'])
        unknows = df.message.tolist()

        if(len(unknows)==0):
            bot.send_message(call.message.chat.id, 'No  Unknown datasets, redirect to menu')
            time.sleep(3)
            restart(call.message)
        else:
            bot.send_message(call.message.chat.id, f'{len(unknows)} Unknown datasets exist')
            max_length = 64 
            
            # oldest record
            x = mycol.find().sort({ "time" : 1 }).limit(1)
            one_unknown = pd.DataFrame(list(x))
            msg = one_unknown['message'].tolist()[0]
            one_unknown = mycol.find_one({'_id':one_unknown['_id'].tolist()[0]})
            print('first query')
            print(one_unknown)
            print(msg)
            if(msg != None):

                indexs,_ = model_process(model,tokenizer,msg,max_length)
                markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)

                for i in range(len(indexs)):
                    found = mongodb_read.query(indexs[i],'original')

                    tag = str(found[0]['tag'])
                    response = found[0]['responses'][0]
                    markup.add(f'{tag} {response}') 


                markup.add('New Topic')
                markup.add('Go to Trash')
                markup.add('Back to menu')
        
                mess = f'Please choose most correspondent answer to question : \n<b>{msg}</b>' 
                mess = f'\nThey are order by most possible responses' 

                msg = bot.send_message(call.message.chat.id, mess, reply_markup=markup, parse_mode='html')
                #send rest of unknown to next func


                bot.register_next_step_handler(msg, insert_delete_unknow,one_unknown)
            else:
                mycol.delete_one({"_id":one_unknown['_id'].tolist()[0]}) 
                update_unknown_datasets(call.message)
    elif call.data == 'statistics_plot':
        try:
            delete_menu(call)
            statistics.show_statistics(call.message)
            time.sleep(10)
            restart(call.message)
        except:
            mess = f'There is something wrong , please contact to fix the problem' 
            msg = bot.send_message(call.message.chat.id,mess)
    else:
        mess = 'probably you did not choose option from buttons, so error occurred'
        bot.send_message(call.message.chat.id,mess)
        time.sleep(10)
        restart(call.message)


# admin section 
        

def admin_confirm(message):

    if message.text == password :
        in_admin.in_admin = True
        admin_start(message)

        ### confirm the accuracy of model everytime admin come

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
    global msg_id
    mess = f'hi, <b>{message.from_user.first_name}</b>!\nWelcome to admin mode'
    #record name and id   
    markup = admin_menu()
    msg_id = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html').message_id
    #bot.register_next_step_handler(msg, force_button_click)

def update_unknown_datasets(message):

    mycol = mongodb_read.mongodb_atlas('unknown_response') 
    unknows = mycol.find()
    unknows = pd.DataFrame(list(unknows))
    unknows = unknows.message.tolist()

    if(len(unknows)==0):
        bot.send_message(message.chat.id, 'No unknown datasets, redirect to menu')
        time.sleep(3)
        admin_start(message)
    else:
        bot.send_message(message.chat.id, f'{len(unknows)} Unknown datasets exist')
        max_length = 64 
         
        x = mycol.find().sort({ "time" : 1 }).limit(1)
        one_unknown = pd.DataFrame(list(x))
        msg = one_unknown['message'].tolist()[0]
        one_unknown = mycol.find_one({'_id':one_unknown['_id'].tolist()[0]})
        print('update_unknown_datasets')
        print(one_unknown)

        if(msg != None):
            indexs,_ = model_process(model,tokenizer,msg,max_length)
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)

            for i in range(len(indexs)):
                found = mongodb_read.query(indexs[i],'original')
                tag = str(found[0]['tag'])
                response = found[0]['responses'][0]
                markup.add(f'{tag} {response}') 

            markup.add('New Topic')
            markup.add('Go to Trash')
            markup.add('Back to menu')

            mess = f'Please choose most correspondent topic to question : \n<b>{msg}</b>' 
            mess = f'\nThey are order by most possible responses' 
            msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')

            #send rest of unknown to next func
            bot.register_next_step_handler(msg, insert_delete_unknow,one_unknown)
        else:
            mycol.delete_one({"_id":one_unknown['_id']}) 
            update_unknown_datasets(message)
def insert_delete_unknow(message,one_unknown):
    msg = message.text
    tag = int(msg.split(' ')[0])
    print(one_unknown)
    try:
        if(msg == 'New Topic'):
            mess = 'please think a response for question'
            msg = bot.send_message(message.chat.id,mess)
            bot.register_next_step_handler(msg, insert_new_topic,one_unknown)
        elif(msg == 'Back to menu'):
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
        else:
            print('before deleted finished in unknown')
            print(one_unknown['_id'])
            print(tag)
            mycol = mongodb_read.mongodb_atlas('unknown_response')
            mycol.delete_one({"_id":one_unknown['_id']}) 
            print('deleted finished in unknown')

            mycol = mongodb_read.mongodb_atlas('training_data')
            found = mycol.find_one({'tag':tag})
            tag = found['tag']
            print(tag)
            one_unknown['response'] = tag
            print('after tag')
            one_unknown.pop('_id')
            

            mycol = mongodb_read.mongodb_atlas('new_response')
            mycol.insert_one(one_unknown) 
            print('insert finished in new')
            mycol = mongodb_read.mongodb_atlas('response_to_user')
            mycol.insert_one(one_unknown) 
            print('insert finished in response to user')

            mess = 'one response updated'
            bot.send_message(message.chat.id,mess)

            time.sleep(3)
            update_unknown_datasets(message)
    except:
        mess = 'probably you did not choose option from buttons, so error occurred'
        bot.send_message(message.chat.id,mess)
        time.sleep(3)
        admin_start(message)

def insert_new_topic(message,one_unknown):
    msg = message.text

    mycol = mongodb_read.mongodb_atlas('unknown_response')
    mycol.delete_one({"_id":one_unknown['_id']}) 

    mycol = mongodb_read.mongodb_atlas('original')

    x= list(mycol.find().sort({ "tag" : -1 }).limit(1))
    label_num = x[0]['tag'] + 1

    dict_example = {'tag':label_num,'patterns':one_unknown['message'],'responses':msg}
    mycol.insert_one(dict_example) 

    one_unknown.pop('_id')
    one_unknown['response'] = label_num
    mycol = mongodb_read.mongodb_atlas('new_response')
    mycol.insert_one(one_unknown) 
        
    mycol = mongodb_read.mongodb_atlas('response_to_user')
    mycol.insert_one(one_unknown) 

    mess = '1 new topic successfully updated'
    bot.send_message(message.chat.id,mess)
    update_unknown_datasets(message)


import time
from telebot import types
import json
import argparse, os
import pandas as pd
import datetime 
import tokens
from NLPmodel import *
from nltk.tokenize import regexp_tokenize
from function import *
from statistics import *

## ----- import pre-trained model

import torch
import pytz
from datetime import datetime

import sys, os
sys.path.append('/opt/homebrew/bin/pip')

eastern_tz = pytz.timezone('Europe/Moscow')

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
        



bot = tokens.bot
myclient = tokens.myclient

class Args():
    embedding_dim = 1024
    hidden=512 
    num_labels = 23
    dropout=0.1
    
args = Args()
PATH = r"itmo_model.pt"
model = QModel_Classifier(args.embedding_dim,args.hidden,args.num_labels,args.dropout)
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)

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
    
    sent = multiple_question_detect(sent)

    model.eval()
    max_length = 64
    print('item num :')
    print(len(sent))
    print()
    print(sent)
    
    model_decode(model,tokenizer,sent,max_length,message)

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
        main_questions_function(call)
    elif call.data == 'more questions':
        msg = bot.send_message(call.message.chat.id, 'please write your questions')
        bot.register_next_step_handler(msg, more_questions)
    elif call.data == 'application':
        linked_user = 'https://signup.itmo.ru/master'
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton(text='redirect to ITMO',
                            url=linked_user))
        mess = 'click to redirect to application form'
        bot.send_message(call.message.chat.id,mess, reply_markup=markup)
        time.sleep(5)
        restart(call.message)

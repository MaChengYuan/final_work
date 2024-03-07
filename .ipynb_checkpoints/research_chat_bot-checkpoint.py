#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
import numpy as np
from transformers import RobertaModel,RobertaTokenizer
class Model_Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels, dropout):
        super(Model_Classifier, self).__init__()
        # Instantiate BERT model
        self.bert = RobertaModel.from_pretrained('roberta-large')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = dropout
        self.linear = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.Drop = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_labels)
        
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            # nn.Dropout(self.dropout),
            #nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0]

        last_hidden_state_cls = self.linear(last_hidden_state_cls)

        last_hidden_state_cls = self.Drop(last_hidden_state_cls)
        

        logits = self.linear2(last_hidden_state_cls)[:, 0, :]

        #logits = self.classifier(last_hidden_state_cls)

        return logits, last_hidden_state_cls,outputs[0]
class QModel_Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels, dropout,feature_remove_max= True):
        super(QModel_Classifier, self).__init__()
        # Instantiate BERT model
        self.bert = RobertaModel.from_pretrained('roberta-large')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = dropout

        
        divisors = sorted(self.cf(embedding_dim,hidden_dim))
        divisors1 = sorted(self.cf(hidden_dim,num_labels))
        common_divisors = sorted(set(divisors1) & set(divisors))
        if(feature_remove_max == True):
            self.n = common_divisors[-1]
        else :
            self.n = common_divisors[0]
        
        self.linear = PHMLayer(self.embedding_dim, self.hidden_dim,self.n)
        self.Drop = nn.Dropout(self.dropout)
        self.linear2 = PHMLayer(self.hidden_dim, self.num_labels,self.n)
        

    def cf(self,num1,num2):
            n=[]
            g=gcd(num1, num2)
            for i in range(1, int(sqrt(g))+1):
                if g%i==0:
                    n.append(i)
                    if g!=i*i:
                        n.append(int(g/i))
            return n

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task

        last_hidden_state_cls = outputs[0]
        #print(last_hidden_state_cls.shape)
        last_hidden_state_cls = self.linear(last_hidden_state_cls)
        #print(last_hidden_state_cls.shape)
        last_hidden_state_cls = self.Drop(last_hidden_state_cls)
        #print(last_hidden_state_cls.shape)

        logits = self.linear2(last_hidden_state_cls)[:, 0, :]
        #print(logits.shape)
        # Feed input to classifier to compute logits
        #logits = self.classifier(last_hidden_state_cls)
        
        return logits, last_hidden_state_cls,outputs[0]


# In[2]:


import telebot
import time
from telebot import types
import json
import torch
import argparse, os

import datetime 
import pymongo

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
        
parser = argparse.ArgumentParser()
#parser.add_argument('-mp','--main_path', help= 'paste path to main QA.json file')
#parser.add_argument('-p','--path', help= 'paste path to test.json file')
#parser.add_argument('-m','--model', help= 'paste path to model file', type=dir_path)

#args = vars(parser.parse_args())
#path_to_main = args["path"]
#path_to_test = args["path"]
#path_to_data = args["model"]


## ----- import pre-trained model

from transformers import RobertaTokenizer
import torch
import pytz
from datetime import datetime
eastern_tz = pytz.timezone('Europe/Moscow')


class Args():
    embedding_dim = 1024
    hidden=512 
    num_labels = 24
    dropout=0.1
    
args = Args()
PATH = r"itmo_model.pt"
model = Model_Classifier(args.embedding_dim,args.hidden,args.num_labels,args.dropout)
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)

record = Record()

token = '6705181314:AAH1F4h1C_rpM5pkcu3tXdeHkznDxIESz3o'
bot = telebot.TeleBot(token, parse_mode='None')
bot_name = 'ITMO BOT'

#model= torch.load('/Users/mac/Desktop/test/model.pth')


def menu():
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton('help', callback_data='help'))   
        markup.add(types.InlineKeyboardButton('contact', callback_data='contact'))   
        markup.add(types.InlineKeyboardButton('main questions', callback_data='main questions'))  
        markup.add(types.InlineKeyboardButton('more questions', callback_data='more questions')) 
        markup.add(types.InlineKeyboardButton('application', callback_data='application'))    
        return markup
def main():       
    @bot.message_handler(commands=['start'])  # Ответ на команду /start
    def start(message):
        mess = f'hi, <b>{message.from_user.first_name}</b>!\nI am - <b>{bot_name}</b>'

        #record name and id 
        record.id = message.chat.id
        record.name = message.from_user.first_name

        
        markup = menu()
        bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')

def restart(message):
    mess = f'hi, <b>{message.from_user.first_name}</b>!\nI am - <b>{bot_name}</b>'
    markup = menu()
    bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')


# In[3]:


with open('main_QA.json', 'r') as json_data:
    main_intents = json.load(json_data)
corpse = []
responses = []
for intent in main_intents['intents']:
    tag = intent['tag']
    response = intent['responses']
    print(tag+"\n")
    corpse.append(tag)# here we are appending the word with its tag
    responses.append(response)


# In[4]:


from nltk.tokenize import regexp_tokenize
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
        mess = 'please choose interested item'
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        for i in range(len(corpse)):            
            markup.add(corpse[i])  
        msg = bot.send_message(call.message.chat.id,mess, reply_markup=markup)
        bot.register_next_step_handler(msg, main_questions)
    elif call.data == 'more questions':
        msg = bot.send_message(call.message.chat.id, 'please write to questions')
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
    


def main_questions(message):
    def tokenize(sentence):
        return regexp_tokenize(sentence, pattern="\w+")

    def score_words(x,y):
          """ returns the jaccard similarity between two lists """
          intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
          union_cardinality = len(set.union(*[set(x), set(y)]))
          return intersection_cardinality/float(union_cardinality)
    sentence = message.text
    if(any(sentence.lower()==item.lower() for item in ["quit","finish","over","bye","goodbye"])):
        print(f"{bot_name}: Goodbye , have a nice day")

    similarity = []
    for i in corpse:
        similarity.append(score_words(sentence,i))
    #print(similarity)
    
    if(max(similarity) > 0.5 and len(tokenize(sentence))==1 ):
        print(f"{bot_name}: "+responses[similarity.index(max(similarity))][0])
    
    mess = responses[similarity.index(max(similarity))][0]
    bot.send_message(message.chat.id, mess)

    time.sleep(5)
    restart(message)
    


# In[5]:


def record_dialogue(record,name):
    token = 'mongodb+srv://mongo:mongo@cluster0.gcj8po2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    myclient = pymongo.MongoClient(token)
    mydb = myclient["itmo_data"]
    mycol = mydb[name]
    #mycol = mydb["customers"]
    now = datetime.now()
    now_russia = eastern_tz.localize(now)            

    mydict = { "name": record.name , "id": record.id, "message": record.message, "predicted":record.predicted, "response":record.response ,"time": now_russia }   
    
    x = mycol.insert_one(mydict)

def query(keylabel):
    
    token = 'mongodb+srv://mongo:mongo@cluster0.gcj8po2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    myclient = pymongo.MongoClient(token)
    
    mydb = myclient["itmo_data"]
    
    mycol = mydb["original"]
    
    myquery = { "tag": keylabel }
    
    mydoc = mycol.find(myquery)
    found = []
    
    for x in mydoc:
      found.append(x)
        
    return found
    


# In[6]:


import spacy
import random
import re
from torch import nn

max_length = 64     


def maximum(list):
    output = list.sort()
    return output[0][-3:] , output[1][-3:]
    #return [output[1][-1],output[1][-2],output[1][-3]]
def multiple_question_detect(sent):
    if(type(sent) == list):
        return sent
    sent = sent.replace('?',' ?')
    sent = re.sub(r'\s\s+',' ',sent)
    sent = re.sub(r"^\s+|\s+$", "", sent)
    sent = re.split('(?<=[.!?,]) +',sent)
    texts = []
    for i in range(len(sent)):
        sent[i] = re.sub('[^a-zA-Z0-9 ]', '', sent[i])
        sent[i] = re.sub(r"^\s+|\s+$", "", sent[i])
        if(len(sent[i])==0):
            texts.append(sent[i])
    
    for i in texts :
        sent.remove(i)
    return sent

def recommendations(message,advice_options):
    
    print('recommendations')
    print(advice_options)

    if(len(advice_options) == 0):
        mess = 'You have reviewed all information'
        mess += '\n'
        mess += 'redirect to main page ... '
        
        bot.send_message(message.chat.id,mess)
        time.sleep(3)
        restart(message)
    elif(len(advice_options) == 1):
        questions =  advice_options
    else:
        questions =  advice_options[:2]
    print(questions)

    
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    
    for index in range(len(questions)):            
        found = query(questions[index])
        
        markup.add(f'''{index} {random.choice(found)['patterns'][0]}''')

    markup.add('None')
    mess = f'Below are options'
    mess += '\n'
    msg = bot.send_message(message.chat.id,mess, reply_markup=markup)
    
    bot.register_next_step_handler(msg, recommendations_decode,advice_options)
        

def recommendations_decode(message,questions):
    if(message.text == 'None'):
        time.sleep(3)
        restart(message)
    else:
        
        questions = questions
        select_index = int(message.text.split(' ')[0])

        # for future RNN recommendation record
        
        record.predicted = None
        record.message = None
        record.response = select_index
        record_dialogue(record,'new_response')
        
        print(questions[select_index])
        
        mess = ''
        found = query(questions[select_index])
        
        mess += random.choice(found)['responses'][0]
        bot.send_message(message.chat.id,mess)
        questions.remove(questions[select_index])
        
        time.sleep(2)  
        mess = 'More recommendations below'
        bot.send_message(message.chat.id,mess)
        recommendations(message,questions)

def model_decode(model ,tokenizer,sents,max_length,message,advice_options = None):
    
    if(len(sents) == 0):
        if(advice_options == None):
            mess = 'redirect to main page'
            bot.send_message(message.chat.id,mess)
            
            time.sleep(5)
            restart(message)
        else:
            mess = f'Here are some related questions that you might be interested'
            mess += '\n'
            
            bot.send_message(message.chat.id,mess)
            recommendations(message,advice_options)
    
            
    else:
        sent = sents[0]
        sents.remove(sent)
        
        print(sent)
        print()


        mess = 'Processing ... (it may takes 5 - 10 seconds)'
        bot.send_message(message.chat.id, mess)
        # encoding and decoding 
        encoding = tokenizer(sent, return_tensors='pt', max_length=max_length, truncation=True)
        b_input_ids = encoding['input_ids']
        #token_type_ids = encoding['token_type_ids']
        attention_mask = encoding['attention_mask']
        outputs = model(b_input_ids, 
                        attention_mask=attention_mask)
        m = nn.Softmax(dim=1)
        outputs= m(outputs[0])
        output_list = outputs[0].detach().numpy()
    
        # first 3 tags
        
        indexs = sorted(range(len(output_list)), key=lambda k: output_list[k], reverse=True)     
        probs = output_list[indexs]
        prob = probs[0]
        index = indexs[0]

        print(indexs)
        print(index)
        
        sents_indexs = []
        sents_indexs.append(sents)
        sents_indexs.append(indexs)
        print(prob)
        print()

        time.sleep(2)
        mess = ''
        if(prob > 0.15):
            found = query(index)

            record.message = sent
            record.predicted = index
            
            mess = random.choice(found)['responses'][0]
            print(mess)
            print()
            bot.send_message(message.chat.id, mess)
        
            #feedback
            time.sleep(5)
            mess = 'is this response answer your questions ?'
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
            markup.add('Yes') 
            markup.add('No') 
            msg = bot.send_message(message.chat.id,mess, reply_markup=markup)
    
            
            
            bot.register_next_step_handler(msg, satisfaction,sents_indexs)
        
        else:
            record.message = sent
            record.predicted = None
            mess = "Sorry I am unable to Process Your Request"
            bot.send_message(message.chat.id, mess)

            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)

            mess = f'belows are possible answers for your questions : {sent}'
            mess += '\n'
            mess += '- - - - - - - - - - - - - - - - - - - - '
            mess += '\n'
            mess += '\n'
            
            for index in range(len(indexs))[:2]:            

                markup.add(str(index))
                
                found = query(indexs[index])

                mess += f'NUMBER {index}.'
                mess += '\n'
                mess += random.choice(found)['responses'][0]
                mess += '\n'
                mess += '\n'

            bot.send_message(message.chat.id,mess)

        
            mess = 'For better performance of system, please click the most correspondent response to your question, thank you for the feedback'    
            markup.add('None') 
            msg = bot.send_message(message.chat.id,mess, reply_markup=markup)
            bot.register_next_step_handler(msg, record_correct_response,sents_indexs)
    


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



def record_correct_response(message,sents_indexs):
    other_answer = sents_indexs[1]
    sents = sents_indexs[0]
    ans = message.text
    if(ans == 'None'):
        
        record.response = None
        record_dialogue(record,'unknown_response')
        
    else:
        record.response = other_answer[int(ans)]
        print('record_correct_response')
        print(record.response)        
        print(other_answer[int(ans)])
        
        #write into database
        
        record_dialogue(record,'new_response')
        
        other_answer.remove(other_answer[int(ans)])
         
    print(record.response)
    print('sents')
    print(sents)

    # redirect to recommeded
    
    redirect_to_model(message,sents,other_answer)



    
def satisfaction(message,sents_indexs):
    
    other_answer = sents_indexs[1]
    print(f'other anwer {other_answer}')
    print(len(other_answer))
    sents = sents_indexs[0]
 
    if(message.text == 'Yes'):
        record.response = other_answer[0]
        other_answer = other_answer[1:]
        print('record.response')        
        print(record.response)
        
        #write into database
        now = datetime.now()
        now_russia = eastern_tz.localize(now)
                
        record.time = now_russia
        record_dialogue(record,'new_response')
        redirect_to_model(message,sents,other_answer)
        

    elif(message.text == 'No'):
        mess = ''
        
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        
        for index in range(len(other_answer))[:2]:            

            markup.add(str(index))
            
            found = query(other_answer[index])
            mess += f'<b>NUMBER {index}.</b>'
            mess += '\n'
            mess += random.choice(found)['responses'][0]
            mess += '\n'
            mess += '\n'
                
                #print(random.choice(intent['responses']))
        bot.send_message(message.chat.id,mess, parse_mode='html')

        
        mess += '\n'
        mess = 'For better performance of system, please click the most correspondent response to your question, thank you for the feedback'    
        markup.add('None') 
        msg = bot.send_message(message.chat.id,mess, reply_markup=markup)
        bot.register_next_step_handler(msg, record_correct_response,sents_indexs)
        
    else:
        mess = ''
        mess = 'I can not understand you, Please follow the instructions'
        mess = '\n'
        mess = 'redirect to main page ...'
        
        msg = bot.send_message(message.chat.id,mess)
        time.sleep(3)
        restart(message)

def redirect_to_model(message,sents,advice_option):
    sents = sents

    time.sleep(5)
    mess = ''
    mess += 'if it is still does not answer your question , please follow instruction below'
    mess += '\n'
    mess += '- - - - - - - - - - - - - - - - - - - - '
    mess += '\n'
    mess += "You may find the way forward in https://en.itmo.ru/en/viewjep/2/5/Big_Data_and_Machine_Learning.htm"
    mess += '\n'
    mess += "Or you may write email to coordinator with aakarabintseva@itmo.ru"
    
    bot.send_message(message.chat.id,mess)

    max_length = 64
    model_decode(model,tokenizer,sents,max_length,message,advice_option)
    
if __name__ == "__main__":
        main()
        bot.polling(none_stop=True)


# In[ ]:





# In[ ]:





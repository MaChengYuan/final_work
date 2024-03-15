# in case for backup

import main
import mongodb_read
import random
import time
import re
import json
from torch import nn
from datetime import datetime
from telebot import types
import tokens
from nltk.tokenize import wordpunct_tokenize
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


with open('/Users/mac/Desktop/SCIENTIFIC_RESEARCH/main_QA.json', 'r') as json_data:
    main_intents = json.load(json_data)
corpse = []
responses = []
for intent in main_intents['intents']:
    tag = intent['tag']
    response = intent['responses']
    print(tag+"\n")
    corpse.append(tag)# here we are appending the word with its tag
    responses.append(response)




corpse = ['requirement','scholarships','batchmates','research topics','internship']
main_index = [0,1,2,3,4,5]
mycol = mongodb_read.mongodb_atlas('training_data')

responses = []
for i in range(len(main_index)):
    x = mycol.find_one({'tag':main_index[i]})
    responses.append(x['responses'])



def main_questions_function(call):
    mess = 'please choose interested item'
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True,resize_keyboard=True)
    for i in range(len(corpse)):            
        markup.add(corpse[i])  
    markup.add('None')  
    try :
        id = call.message.chat.id
    except :
        id = call.chat.id

    msg = main.bot.send_message(id,mess, reply_markup=markup)
    main.bot.register_next_step_handler(msg, main_questions)

def main_questions(message):
    print('hi')
    mess = None
    if(message.text == 'None'):
        mess = 'redirecting to menu ...'
        main.bot.send_message(message.chat.id, mess,reply_markup=types.ReplyKeyboardRemove())

        time.sleep(3)
        main.restart(message)
    else:  
        if(message.text in corpse):
            msg = responses[corpse.index(message.text)]
            print(f"{main.bot_name}: "+msg)
            mess = msg
        else:
            msg = 'You must click one of the options!'
            print(f"{main.bot_name}: "+msg)
            mess = msg
    
        
        main.bot.send_message(message.chat.id, mess,reply_markup=types.ReplyKeyboardRemove())
    
        time.sleep(5)
        main_questions_function(message)
    


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
        
        main.bot.send_message(message.chat.id,mess)
        time.sleep(3)
        main.restart(message)
    elif(len(advice_options) == 1):
        questions =  advice_options
    else:
        questions =  advice_options[:2]
    print(questions)

    
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    
    for index in range(len(questions)):            
        found = mongodb_read.query(questions[index])
        
        markup.add(f'''{index}. {random.choice(found)['patterns'][0]}''')
    markup.add('I want to write')
    markup.add('None')

    mess = f'Below are options'
    mess += '\n'
    msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
    
    main.bot.register_next_step_handler(msg, recommendations_decode,advice_options)
        

def recommendations_decode(message,questions):
    if(message.text == 'None'):
        mess = 'redirect to main page ... '      
        main.bot.send_message(message.chat.id,mess)
        time.sleep(3)
        main.restart(message)
    elif(message.text == 'I want to write'):
        msg = main.bot.send_message(message.chat.id, 'please write your questions')
        main.bot.register_next_step_handler(msg, main.more_questions)
    else:     
        questions = questions
        print(message.text)
        select_index = int(message.text.split('.')[0])

        # for future RNN recommendation record
        
        main.record.predicted = None
        main.record.message = None
        main.record.response = select_index
        mongodb_read.record_dialogue(main.record,'new_response')
        
        print(questions[select_index])
        
        mess = ''
        found = mongodb_read.query(questions[select_index])
        
        mess += random.choice(found)['responses'][0]
        main.bot.send_message(message.chat.id,mess)
        questions.remove(questions[select_index])
        
        time.sleep(2)  
        mess = 'More recommendations below'
        main.bot.send_message(message.chat.id,mess)
        recommendations(message,questions)

def model_decode(model ,tokenizer,sents,max_length,message,advice_options = None):
    
    if(len(sents) == 0):
        if(advice_options == None):
            mess = 'redirect to main page'
            main.bot.send_message(message.chat.id,mess)
            
            time.sleep(5)
            main.restart(message)
        else:
            mess = f'Here are some related questions that you might be interested'
            mess += '\n'
            
            main.bot.send_message(message.chat.id,mess)
            recommendations(message,advice_options)            
    else:
        sent = sents[0]
        sents.remove(sent)
        
        print(sent)
        print()
        
        mess = 'Processing ... (it may takes 5 - 10 seconds)'
        main.bot.send_message(message.chat.id, mess)
        # encoding and decoding 
        
        indexs , probs = main.model_process(main.model,main.tokenizer,sent,max_length)
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
        if(prob > 0.85):
            found = mongodb_read.query(index)
            main.record.message = sent
            main.record.predicted = index

            mess = f'To question : <b>{sent}</b>'
            mess = '\n'
            mess = random.choice(found)['responses'][0]
            print(mess)
            print()
            main.bot.send_message(message.chat.id, mess, parse_mode='html')
        
            #feedback
            time.sleep(5)
            mess = 'is this response answer your questions ?'
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
            markup.add('Yes') 
            markup.add('No') 
            msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
    
            
            
            main.bot.register_next_step_handler(msg, satisfaction,sents_indexs)
        
        else:
            main.record.message = sent
            main.record.predicted = None
            mess = "Sorry I am unable to Process Your Request"
            main.bot.send_message(message.chat.id, mess)

            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)

            mess = f'belows are possible answers for your questions : {sent}'
            mess += '\n'
            mess += '- - - - - - - - - - - - - - - - - - - - '
            mess += '\n'
            mess += '\n'
            
            for index in range(len(indexs))[:2]:            

                markup.add(str(index))
                
                found = mongodb_read.query(indexs[index])

                mess += f'NUMBER {index}.'
                mess += '\n'
                mess += random.choice(found)['responses'][0]
                mess += '\n'
                mess += '\n'

            main.bot.send_message(message.chat.id,mess)

        
            mess = 'For better performance of system, please click the most correspondent response to your question, thank you for the feedback'    
            markup.add('None') 
            msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
            main.bot.register_next_step_handler(msg, record_correct_response,sents_indexs)
    


def detect_meaningless_sentence(str):
    def clean_punctuation(string):
        try:
            res = re.sub(r'[^\w\s]', '', string)
            return res
        except:
            return string
            
    x = clean_punctuation(str)
    x = wordpunct_tokenize(x)
    x = [w for w in x if not w.lower() in stop_words]

    return x


def record_correct_response(message,sents_other_answer):
    other_answer = sents_other_answer[1]
    sents = sents_other_answer[0]
    ans = message.text

    detect_tokens = detect_meaningless_sentence(sents)
    if(detect_tokens < 1 ):
        main.record.response = None
        mongodb_read.record_dialogue(main.record,'trash')
        redirect_to_model(message,sents,other_answer)
    else:
        if(ans == 'None'):
            
            main.record.response = None
            mongodb_read.record_dialogue(main.record,'unknown_response')
            
        else:
            main.record.response = other_answer[int(ans)]
            
            #write into database
            
            mongodb_read.record_dialogue(main.record,'new_response')
            
            other_answer.remove(other_answer[int(ans)])
            
        # redirect to recommeded
        
        redirect_to_model(message,sents,other_answer)



    
def satisfaction(message,sents_other_answer):
    
    other_answer = sents_other_answer[1]
    print(f'other anwer {other_answer}')
    print(len(other_answer))
    sents = sents_other_answer[0]
    now = datetime.now()
    now_russia = tokens.eastern_tz.localize(now)
    main.record.time = now_russia

    if(message.text == 'Yes'):
        main.record.response = other_answer[0]
        other_answer = other_answer[1:]
        #write into database

        detect_tokens = detect_meaningless_sentence(sents)

        if(detect_tokens < 1 ):
            mongodb_read.record_dialogue(main.record,'trash')
            redirect_to_model(message,sents,other_answer)
        else:
            mongodb_read.record_dialogue(main.record,'new_response')
            redirect_to_model(message,sents,other_answer)
            

    elif(message.text == 'No'):
        first_to_last = other_answer[0]
        other_answer = other_answer[1:]
        other_answer.append(first_to_last)
        sents_other_answer[1] = other_answer

        mess = ''
        
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        
        for index in range(len(other_answer))[:2]:            

            markup.add(str(index))
            
            found = mongodb_read.query(other_answer[index])
            mess += f'<b>No. {index}.</b>'
            mess += '\n'
            mess += random.choice(found)['responses'][0]
            mess += '\n'
            mess += '\n'
                
                #print(random.choice(intent['responses']))
        main.bot.send_message(message.chat.id,mess, parse_mode='html')

        
        mess += '\n'
        mess = 'For better performance of system, please click the most correspondent response to your question, thank you for the feedback'    
        markup.add('None') 
        msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
        main.bot.register_next_step_handler(msg, record_correct_response,sents_other_answer)
        
    else:
        mess = ''
        mess = 'I can not understand you, Please follow the instructions'
        mess = '\n'
        mess = 'redirect to main page ...'
        
        msg = main.bot.send_message(message.chat.id,mess)
        time.sleep(3)
        main.restart(message)

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
    
    main.bot.send_message(message.chat.id,mess)

    time.sleep(5)
    max_length = 64
    model_decode(main.model,main.tokenizer,sents,max_length,message,advice_option)
    

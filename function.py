# in case for backup

import main
import mongodb_read
import time
import re
from datetime import datetime
from telebot import types
from nltk.tokenize import wordpunct_tokenize
import nltk
from nltk.corpus import stopwords
import recommend as recommend 
import os
from copy import deepcopy
 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class query_histroy():
    def __init__(self):
        self.current_query_histroy = []

current_query_histroy = query_histroy()

        
# to force users to click button
def force_button_click(message):
    if not message.text.startswith('/'):
        main.bot.send_message(message.chat.id, 'You must click one of the buttons!')
        
    time.sleep(3)
    main.restart(message)

corpse = ['requirement','scholarships','batchmates','research topics','internship']
main_index = [0,1,2,4,5]
mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')

responses = []
for i in range(len(main_index)):
    x = mycol.find_one({'tag':main_index[i]})
    responses.append(x['responses'])



def main_questions_function(call):
    mess = 'please choose interested item'
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True,resize_keyboard=True)
    for i in range(len(corpse)):            
        markup.add(corpse[i])  
    markup.add('Back to menu')  
    try :
        id = call.message.chat.id
    except :
        id = call.chat.id

    msg = main.bot.send_message(id,mess, reply_markup=markup)
    main.bot.register_next_step_handler(msg, main_questions)

def main_questions(message):
    mess = None

    if(message.text == 'Back to menu'):
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

def sep_by_and(last_sent):
    sent = []
    for i in last_sent : 
        sent.extend(i.split('and'))
    texts = []
    for i in range(len(sent)):
    
        sent[i] = re.sub('[^a-zA-Z0-9 ]', '', sent[i])
        sent[i] = re.sub(r"^\s+|\s+$", "", sent[i])
        if(len(sent[i])==0):
            texts.append(sent[i])
    
    for i in texts :
        sent.remove(i)
    return sent

def SASRec_recommendations(message):
    print('here is SASRec_recommendations')
    config = recommend.Args()
    print(f'history : {current_query_histroy.current_query_histroy}')
    print(len(list(set(current_query_histroy.current_query_histroy))))
    print(config.item_id_max - len(list(set(current_query_histroy.current_query_histroy))))
    if(config.item_id_max - len(list(set(current_query_histroy.current_query_histroy)))==0):
        mess = 'You have reviewed all information, Scroll up to review responses or Go to Write question sections'
        mess += '\n'
        mess += 'redirect to main page ... '
        
        main.bot.send_message(message.chat.id,mess,reply_markup=types.ReplyKeyboardRemove())
        time.sleep(3)
        main.restart(message)
    else:
        current_query_histroy_copy = deepcopy(current_query_histroy.current_query_histroy)
        questions, _ = recommend.predict(config,current_query_histroy_copy)
        questions = questions[0]
        
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        try:
            questions = questions[:2]
        except:
            # only one options left
            pass
        questions = questions.tolist()
        
        print(f'options {questions}')
        for index in range(len(questions)):     
            mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')  
            markup.add(f'''{index}. {mycol.find_one({'tag':questions[index]-1})['patterns']}''')   
            # found = mongodb_read.query(questions[index]-1,'original')
            
            # markup.add(f'''{index}. {random.choice(found)['patterns'][0]}''')

        markup.add('I want to write')
        markup.add('None')

        mess = f'Below are options'
        mess += '\n'
        msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
        
        questions_srs = []
        questions_srs.append(questions)
        srs = True
        questions_srs.append(srs)

        main.bot.register_next_step_handler(msg, recommendations_decode,questions_srs)
            


def recommendations(message,advice_options):
    
    print('recommendations')
    print(advice_options)

    if(len(advice_options) == 0):
        mess = 'You have reviewed all information, Scroll up to review responses or Go to Write question sections'
        mess += '\n'
        mess += 'redirect to main page ... '
        
        main.bot.send_message(message.chat.id,mess,reply_markup=types.ReplyKeyboardRemove())
        time.sleep(3)
        main.restart(message)
    elif(len(advice_options) == 1):
        questions =  advice_options
    else:
        questions =  advice_options[:2]
    print(questions)
    
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    
    for index in range(len(questions)):            
        mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')  
        markup.add(f'''{index}. {mycol.find_one({'tag':questions[index]})['patterns']  }''')  
        # found = mongodb_read.query(questions[index],'original')
        
        # markup.add(f'''{index}. {random.choice(found)['patterns'][0]}''')
    markup.add('I want to write')
    markup.add('None')

    mess = f'Below are options'
    mess += '\n'
    msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
    
    questions_srs = []
    questions_srs.append(advice_options)
    srs = False
    questions_srs.append(srs)

    main.bot.register_next_step_handler(msg, recommendations_decode,questions_srs)
        

def recommendations_decode(message,questions_srs):

        if(message.text == 'None'):
            mess = 'redirect to main page ... '      
            main.bot.send_message(message.chat.id,mess,reply_markup=types.ReplyKeyboardRemove())
            time.sleep(3)
            main.restart(message)
        elif(message.text == 'I want to write'):
            db_list = mongodb_read.show_program_name('itmo_data',True)
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
            for name in db_list:
                markup.add(f'{name}') 
            mess = 'Please choose target program'
            msg = main.bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
            main.bot.register_next_step_handler(msg, main.choose_program)
        else:    
            select_index = int(message.text.split('.')[0]) 
            questions = questions_srs[0]
            srs =  questions_srs[1]
            print(questions[select_index])
            print(select_index)
            current_query_histroy.current_query_histroy.append(questions[select_index])
            print('here is recommendations_decode')
            print(f'history : {current_query_histroy.current_query_histroy}')
        
            if(srs == True):
                main.record.predicted = None
                main.record.message = None
                main.record.response = select_index
                main.record.modified_response = None
                
                mongodb_read.record_dialogue('global',main.record,'new_response')

                mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')  
                mess = mycol.find_one({'tag':questions[select_index]-1})['responses']
                main.bot.send_message(message.chat.id,mess)
                
                time.sleep(2)  
                mess = 'More recommendations below'
                main.bot.send_message(message.chat.id,mess)
                SASRec_recommendations(message)

            else:
                questions = questions
                print(message.text)

                main.record.predicted = None
                main.record.message = None
                main.record.response = select_index
                main.record.modified_response = None
                mongodb_read.record_dialogue('global',main.record,'new_response')
                
                print(questions[select_index])
                
                mess = ''
                mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')  
                mess += mycol.find_one({'tag':questions[select_index]})['responses']
                # found = mongodb_read.query(questions[select_index],'original')                
                # mess += random.choice(found)['responses'][0]

                main.bot.send_message(message.chat.id,mess)
                questions.remove(questions[select_index])
                
                time.sleep(2)  
                mess = 'More recommendations below'
                main.bot.send_message(message.chat.id,mess)
                recommendations(message,questions)

def model_decode(sents,max_length,message,advice_options = None):
    
    if(len(sents) == 0):
        # choice of two differnt recommendation system by senting rest of options or no
        if(advice_options == None):
            print('i am sasrec')
            mess = f'Here are some related questions that you might be interested'
            mess += '\n'
            main.bot.send_message(message.chat.id,mess)
            SASRec_recommendations(message)
        else:
            print('i am NLPrec')
            mess = f'Here are some related questions that you might be interested'
            mess += '\n'
            
            main.bot.send_message(message.chat.id,mess)
            recommendations(message,advice_options)            
    else:

        sent = sents[0]
        
        mess = 'Processing ... (it may takes 5 - 10 seconds)'
        main.msg_id = main.bot.send_message(message.chat.id, mess).message_id
        # encoding and decoding 
        print('before process')
        print(sent)
        print()
        indexs , probs = main.model_process(main.model,main.tokenizer,sent,max_length)
        prob = probs[0]
        index = indexs[0]

        print(indexs)
        print(index)
        
        sents_other_answer_options = []
        sents_other_answer_options.append(sents)
        sents_other_answer_options.append(indexs)
        print(prob)
        print()

        time.sleep(2)
        mess = ''
        if(prob > 0.75):
            mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')  
            
            #found = mongodb_read.query(index,'original')
            main.record.message = sent.split(':')[1]
            main.record.predicted = index

            mess = f'To question : <b>{sent}</b>'
            mess = '\n'
            mess = mycol.find_one({'tag':index})['responses']
            #mess = random.choice(found)['responses'][0]
            print(mess)
            print()

            main.message_delete_menu(message)
            main.bot.send_message(message.chat.id, mess, parse_mode='html')
        
            #feedback
            time.sleep(5)
            mess = 'is this response answer your questions ?'
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
            markup.add('Yes') 
            markup.add('No') 
            msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)

            main.bot.register_next_step_handler(msg, satisfaction,sents_other_answer_options) 
        else:
            main.record.message = sent.split(':')[1]
            main.record.predicted = None
            mess = '- - - - - - - - - - - - - - - - - - - - '
            mess += '\n'
            mess += f"Sorry I am unable to Process Your Request : {sent}"
            main.bot.send_message(message.chat.id, mess)

            mess = f'belows are possible answers for your questions'
            mess += '\n'
            mess += '- - - - - - - - - - - - - - - - - - - - '
            mess += '\n'
            mess += '\n'

            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
            options = []
            for index in range(len(indexs))[:2]:            
                markup.add(str(index))
                options.append(index)
                mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')  
                #found = mongodb_read.query(indexs[index],'original')
                mess += f'NO {index}.'
                mess += '\n'
                mess += mycol.find_one({'tag':indexs[index]})['responses']
                #mess += random.choice(found)['responses'][0]
                mess += '\n'
                mess += '\n'
            markup.add('More') 
            markup.add('None') 
            sents_other_answer_options.append(options)

            main.bot.send_message(message.chat.id,mess)
            mess = 'For better performance of system, please click the most correspondent response to your question, thank you for the feedback'    
            msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
            main.bot.register_next_step_handler(msg, record_correct_response,sents_other_answer_options)
    


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


def record_correct_response(message,sents_other_answer_options):
    other_answer = sents_other_answer_options[1]
    sents = sents_other_answer_options[0]
    options = sents_other_answer_options[2]
    sent = sents[0]
    ans = message.text

    try:
        detect_tokens = detect_meaningless_sentence(sent)
        if(len(detect_tokens) < 1 ):
            #main.record.response = None
            #mongodb_read.record_dialogue(main.record,'trash')
            print('it is garbage')
            redirect_to_model(message,sents,other_answer)
        else:
            print('i am here')
            if(ans == 'None'):
                
                main.record.response = None
                main.record.modified_response = None
                mongodb_read.record_dialogue('global',main.record,'unknown_response')
                #delete current questions 
                sents.remove(sent)
                redirect_to_model(message,sents,other_answer)
            elif(ans == 'More'):
                first_two = other_answer[:2]
                other_answer = other_answer[2:]
                other_answer.extend(first_two)
                markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
                options = []
                mess = ''
                for index in range(len(other_answer))[:2]:            
                    markup.add(str(index))
                    options.append(index)
                    mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')  
                    #found = mongodb_read.query(indexs[index],'original')
                    mess += f'NO {index}.'
                    mess += '\n'
                    mess += mycol.find_one({'tag':other_answer[index]})['responses']
                    #mess += random.choice(found)['responses'][0]
                    mess += '\n'
                    mess += '\n'
                markup.add('More') 
                markup.add('None') 
                sents_other_answer_options[2] = options
                sents_other_answer_options[1] = other_answer
                print('after change')
                print(other_answer)
                print(options)

                main.bot.send_message(message.chat.id,mess)
                mess = 'For better performance of system, please click the most correspondent response to your question, thank you for the feedback'    
                msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
                main.bot.register_next_step_handler(msg, record_correct_response,sents_other_answer_options)

            elif(int(ans) in options):
                #pluse because train sasrec model with index + 1
                current_query_histroy.current_query_histroy.append(other_answer[int(ans)]+1)
                print('here is record_correct_response')
                print(f'history : {current_query_histroy.current_query_histroy}')

                main.record.response = other_answer[int(ans)]
                
                #write into database
                main.record.modified_response = None
                mongodb_read.record_dialogue('global',main.record,'new_response')

                print(other_answer[int(ans)])
                print(ans)
                other_answer.remove(other_answer[int(ans)]) 
                #delete current questions 
                sents.remove(sent)
                redirect_to_model(message,sents,other_answer)
    except:
        
        mess = 'probably you did not choose option from buttons, so error occurred'
        mess += '\n'
        mess += 'redirect to main page ...'
        main.bot.send_message(message.chat.id,mess,reply_markup=types.ReplyKeyboardRemove())
        time.sleep(3)
        main.restart(message)


    
def satisfaction(message,sents_other_answer):


    other_answer = sents_other_answer[1]
    print(f'other anwer {other_answer}')
    print(len(other_answer))
    sents = sents_other_answer[0]

    now = datetime.now()    
  
    main.record.time = now

    if(message.text == 'Yes'):
        main.record.response = other_answer[0]

        #plus 1 because train sasrec model with index + 1
        current_query_histroy.current_query_histroy.append(other_answer[0]+1)
        print('here is satisfaction')
        print(f'history : {current_query_histroy.current_query_histroy}')

        other_answer = other_answer[1:]
        #write into database
        sent = sents[0]
        sents.remove(sent)
        
        print(sent)
        detect_tokens = detect_meaningless_sentence(sent)

        if(len(detect_tokens) < 1 ):
            print('it is garbage')
            #mongodb_read.record_dialogue(main.record,'trash')
            redirect_to_model(message,sents,other_answer)
        else:
            main.record.modified_response = None
            mongodb_read.record_dialogue('global',main.record,'new_response')
            redirect_to_model(message,sents,other_answer)
    

    elif(message.text == 'No'):
        first_to_last = other_answer[0]
        other_answer = other_answer[1:]
        other_answer.append(first_to_last)
        sents_other_answer[1] = other_answer

        mess = ''
        
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        options = []
        for index in range(len(other_answer))[:2]:            
            options.append(index)
            markup.add(str(index))
            mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')  
            #found = mongodb_read.query(other_answer[index],'original')
            mess += f'<b>No. {index}.</b>'
            mess += '\n'
            mess += mycol.find_one({'tag':other_answer[index]})['responses']
            #mess += random.choice(found)['responses'][0]
            mess += '\n'
            mess += '\n'
                
                #print(random.choice(intent['responses']))
        main.bot.send_message(message.chat.id,mess, parse_mode='html').message_id

        
        mess += '\n'
        mess = 'For better performance of system, please click the most correspondent response to your question, thank you for the feedback'    
        markup.add('More') 
        markup.add('None') 
        msg = main.bot.send_message(message.chat.id,mess, reply_markup=markup)
        sents_other_answer_options=[]
        sents_other_answer_options.append(sents)
        sents_other_answer_options.append(other_answer)
        sents_other_answer_options.append(options)
        main.bot.register_next_step_handler(msg, record_correct_response,sents_other_answer_options)
        
    else:
        
        mess = 'I can not understand you, Please click the button'
        mess += '\n'
        mess += 'redirect to main page ...'
        
        main.msg_id = main.bot.send_message(message.chat.id,mess,reply_markup=types.ReplyKeyboardRemove()).message_id
        time.sleep(3)
        main.restart(message)

def redirect_to_model(message,sents,advice_option):
    sents = sents

    time.sleep(5)
    mess = ''
    mess += 'if it is still does not answer your question perfectly , please follow instruction below'
    mess += '\n'
    mess += '- - - - - - - - - - - - - - - - - - - - '
    mess += '\n'
    mess += "You may find the way forward in https://en.itmo.ru/en/viewjep/2/5/Big_Data_and_Machine_Learning.htm"
    mess += '\n'
    mess += "Or you may write email to coordinator with aakarabintseva@itmo.ru"
    
    main.bot.send_message(message.chat.id,mess)

    time.sleep(5)
    max_length = 64

    if(os.path.exists('recommendation.pth')):
        print('path exists')
        model_decode(sents,max_length,message)
    else:
        model_decode(sents,max_length,message,advice_option)
    

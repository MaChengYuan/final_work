import time
import threading
from telebot import types
import telebot
import os
from torch import nn
import mongodb_read 
import statistics_1 
from Train import NLPmodel 
import function 
import pandas as pd
import torch
import speech_recognition as sr
import subprocess
import json

from tokens import token , password
import retrain


class admin_mode_or_no():
    def __init__(self):
        """
        :param args:
        """
        super(admin_mode_or_no, self).__init__()
        self.in_admin = False

in_admin = admin_mode_or_no()

# bot instrction menu
bot = telebot.TeleBot(token, parse_mode='None')
# c1 = types.BotCommand(command='start', description='Start the Bot')
# bot.set_my_commands([c1])
# bot.set_chat_menu_button(bot.get_me().id, types.MenuButtonCommands('commands'))


@bot.message_handler(commands=['admin'])  
def admin(message):
    message_delete_menu(message)
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
        self.program = None
        self.message = None
        self.predicted = None
        self.response = None
        self.time = None
        self.modified_response = None

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        

class Args():
    def __init__(self):
        self.embedding_dim = 1024
        self.hidden=512 
        # mycol = mongodb_read.mongodb_atlas('global','train_data')
        # self.num_labels = list(mycol.find().sort('tag'))[-1]['tag']+1

        mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')
        self.mycol = mongodb_read.mongodb_atlas('global','train_data')
        self.num_labels = list(mycol.find().sort('tag'))[-1]['tag']+1
        self.dropout=0.1
    
args = Args()

PATH = r"itmo_model.pt"

file_exist = os.path.isfile(PATH)

model = NLPmodel.QModel_Classifier(args.embedding_dim,args.hidden,args.num_labels,args.dropout)
tokenizer = NLPmodel.RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)

record = Record()
bot_name = 'ITMO BOT'

msg_id = None

try:
    if(file_exist):
        model.load_state_dict(torch.load(PATH))
        model.eval()
    else:
        print('Maintanance')
except:
    file_exist = False
    print('Maintanance')

def menu():
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton('Help', callback_data='help'))   
        markup.add(types.InlineKeyboardButton('Contact', callback_data='contact'))   
        markup.add(types.InlineKeyboardButton('Main questions', callback_data='main questions'))  
        markup.add(types.InlineKeyboardButton('Write questions', callback_data='more questions')) 
        markup.add(types.InlineKeyboardButton('Application', callback_data='application')) 
        markup.add(types.InlineKeyboardButton('test', callback_data='test')) 


        #markup.add(types.InlineKeyboardButton('audio_test', callback_data='audio_test'))    

        return markup



#@bot.message_handler(content_types=['text'])
def start(message):
    mess = 'Hi !!!'
    bot.send_message(message.chat.id, mess)

    mess = 'It seems like you do not have "start" button in your telegram\nPlease type /start manually to go to main page'
    bot.send_message(message.chat.id, mess)

@bot.message_handler(commands=['start']) 
def start(message):
    try:
        # admin model false
        in_admin.in_admin = False
        mess = f'hi, <b>{message.from_user.first_name}</b>!\nI am - <b>{bot_name}</b>'
        mess += f'\n'

        #detect if there is response from staff    
        mycol = mongodb_read.mongodb_atlas('global','response_to_user')
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
                mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')
                mess += mycol.find_one({'tag':list_of_res[i]})['responses']
                # found = mongodb_read.query(list_of_res[i],'original')
                # mess += random.choice(found)['responses'][0]
            
        #record name and id 
        record.id = message.chat.id
        record.name = message.from_user.first_name
        
        markup = menu()

        global msg_id
        msg_id = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html').message_id
    except:
        mess = 'please click buttons'
        bot.send_message(message.chat.id, mess)
        time.sleep(2)
        restart(message)
    


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

def recognize_speech(message):
    try:
        print('recognize_speech')
        recognizer = sr.Recognizer()
        info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(info.file_path)
        src_filename = 'user_voice.ogg'
        with open(src_filename, 'wb') as new_file:
            new_file.write(downloaded_file)

        dest_filename = 'output.wav'
        process = subprocess.run(['ffmpeg', '-y', '-i', src_filename, dest_filename])
        if process.returncode != 0:
            mess = "Sorry, there was an error processing your request.\n"
            mess += "Please write your questions with text."

            bot.send_message(message.chat.id, mess)
            #raise Exception("Something went wrong")
            
        print('before read audio')
        user_audio_file = sr.AudioFile(dest_filename)
        print('after read audio')
        with user_audio_file as source:
            user_audio = recognizer.record(source)
        text = recognizer.recognize_google(user_audio, language='en-US')
            
        mess = f"You said: {text}"
        print(mess)
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
        markup.add('Yes')
        markup.add('Record again')
        markup.add('Back to menu')

        mess = f'Is this what you just said ? : \n<b>{text}</b>' 
        msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
        bot.register_next_step_handler(msg, check_audio,text)
    except sr.UnknownValueError:
        mess = "Sorry, I couldn't understand that."
        mess = 'please say it loudly or <b>clearly</b> write your questions \nIf multiple questions, please separate by <b>. or and</b>'
        msg = bot.send_message(message.chat.id, mess,parse_mode='HTML')
        bot.register_next_step_handler(msg, more_questions)
    except:

        mess = "Sorry, there was an error processing your request.\n"
        mess += "Please write your questions with text."
        #for debug
        print('second error')
        bot.send_message(message.chat.id, mess)
        time.sleep(5)
        restart(message)


def check_audio(message,text):
    msg = message.text
    print('check_audio')
    if(msg == 'Yes'):
        if(text == None):
            mess = "Sorry, there was an error processing your request.\n"
            mess += "Please write your questions with text."
            #for debug
            print('second error')
            bot.send_message(message.chat.id, mess)
            time.sleep(5)
            restart(message)
        else:
            more_questions(message,text)
    elif(msg == 'Record again'):
        mess = 'please <b>clearly</b> write your questions \nIf multiple questions, please separate by <b>. or and</b>'
        msg = bot.send_message(message.chat.id, mess,parse_mode='HTML')
        bot.register_next_step_handler(msg, more_questions)
    elif(msg == 'Back to menu'):
        time.sleep(5)
        restart(message)  
    else:
        mess = "Something went wrong , back to menu"
        bot.send_message(message.chat.id, mess)
        time.sleep(5)
        restart(message)
 
def more_questions(message,program_audio_text = None):
    try:
        program = program_audio_text[0]
        audio_text = program_audio_text[1]
        record.program = program
        print('more_questions')
        if(audio_text != None ):
            sent = audio_text
        else:
            sent = message.text
        
        sent = sent.lower()
        sent = function.multiple_question_detect(sent)
        sent = function.sep_by_and(sent)

        max_length = 64
        print('item num :')
        print(len(sent))
        print()
        print(sent)
        sent[0] = f'{program}:' + sent[0]

        function.model_decode(sent,max_length,message)

    except:
        print('voice')
        recognize_speech(message)
        

def call_delete_menu(call):
    global msg_id
    bot.delete_message(call.message.chat.id, msg_id)
    msg_id = None

def message_delete_menu(message):
    global msg_id

    bot.delete_message(message.chat.id, msg_id)
    msg_id = None

@bot.callback_query_handler(func=lambda call: True)
def message_reply(call):
    if call.data == 'contact':
        call_delete_menu(call)
        mess = 'click on mail to get contact with staff'
        bot.send_message(call.message.chat.id,mess)

        # to delete last message
        # try <a href='aakarabintseva@itmo.ru'>Program coordinator</a>
        mess = """
1. aakarabintseva@itmo.ru :  Program coordinator

2. international@itmo.ru  :  International office 

3. int.students@itmo.ru   :  Student office 

4. aakhalilova@itmo.ru    :  Migration office
"""
        bot.send_message(call.message.chat.id,mess, parse_mode='html')
        time.sleep(5)
        restart(call.message)
    elif call.data == 'help': 
        call_delete_menu(call)
        mess = """
Contact --  to find Email address of specific staff in ITMO
Main questions -- to answer most frequent questions from candidates
Write questions -- to write your own question with text
Application -- to redirect to page to fill application
"""
        bot.send_message(call.message.chat.id,mess)
        # to delete last message

        time.sleep(5)
        restart(call.message)
    elif call.data == 'main questions':
        call_delete_menu(call)
        function.main_questions_function(call)
    elif call.data == 'more questions': 
        call_delete_menu(call)
        if(file_exist): 
            db_list = mongodb_read.show_program_name('itmo_data',True)

            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)

            for name in db_list:
                markup.add(f'{name}') 
            mess = 'Please choose target program'
            msg = bot.send_message(call.message.chat.id, mess, reply_markup=markup, parse_mode='html')
            bot.register_next_step_handler(msg,choose_program)
        else:
            mess = f'The system is under maintenance, sorry for inconvenience'
            bot.send_message(call.message.chat.id, mess)
            time.sleep(5)
            restart(call.message)

    elif call.data == 'application':
        call_delete_menu(call)
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
        call_delete_menu(call)
        mycol = mongodb_read.mongodb_atlas('global','unknown_response') 
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

            print('first query')
            print(one_unknown)
            print(msg)
            if(msg != None):

                indexs,_ = model_process(model,tokenizer,msg,max_length)
                markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)

                for i in range(len(indexs)):
                    mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning') 
                    found = mycol.find_one({'tag':indexs[i]}) 
                    tag = str(found['tag'])
                    response = found['responses']
                    markup.add(f'{tag} {response}') 

                markup.add('New Topic')
                markup.add('Go to Trash')
                markup.add('Back to menu')
        
                mess = f"""Please choose most correspondent answer to question : \n<b>{msg}</b> about {one_unknown['program'].tolist()[0]}""" 
                mess += f'\n\nThey are order from most possible responses to least ones' 

                msg = bot.send_message(call.message.chat.id, mess, reply_markup=markup, parse_mode='html')
                #send rest of unknown to next func


                bot.register_next_step_handler(msg, insert_delete_unknow,one_unknown)
            else:
                mycol.delete_one({"_id":one_unknown['_id'].tolist()[0]}) 
                update_unknown_datasets(call.message)
    elif call.data == 'Update_responses':
        mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')
        x = mycol.find().sort('tag')
        train_data = pd.DataFrame(list(x))[['tag','responses']].drop_duplicates()
        tags = train_data.tag.tolist()
        responses = train_data.responses.tolist()
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        for i in range(len(tags)):
            markup.add(f'{tags[i]} {responses[i]}') 

        mess = f'Please choose text which you want to correct' 
        msg = bot.send_message(call.message.chat.id, mess, reply_markup=markup, parse_mode='html')
        bot.register_next_step_handler(msg, update_responses)


    elif call.data == 'statistics_plot':
        try:
            call_delete_menu(call)
            statistics_1.show_statistics(call.message)
            time.sleep(10)
            admin_start(call.message)
        except:
            mess = f'There is something wrong , please contact to fix the problem' 
            msg = bot.send_message(call.message.chat.id,mess)

    # change_password
    elif call.data == 'change_password':
        call_delete_menu(call)
        bot.register_next_step_handler(msg, change_password)

    elif call.data == 'add new program':
        mess = f'Please send files'
        mess += f'File name must be program name, for example : computer science.json'
        mess += f'\nFile must be json file and\nFormat should be "patterns" with questions and "responses" with responses' 

        msg = bot.send_message(call.message.chat.id, mess)
        bot.register_next_step_handler(msg, insert_new_program_info)

    elif call.data == 'test':
        call_delete_menu(call)
        import threading
        t1=threading.Thread(target=retrain.train_process)
        t1.start()
        time.sleep(10)
        restart(call.message)
        
    else:
        call_delete_menu(call)
        mess = 'probably you did not choose option from buttons, so error occurred'
        bot.send_message(call.message.chat.id,mess)
        time.sleep(10)
        restart(call.message)

def choose_program(message):
    db_list = mongodb_read.show_program_name()
    program = message.text
    if(program in db_list):
        mess = 'please <b>clearly</b> write your questions \nIf multiple questions, please separate by <b>. or and</b>'
        msg = bot.send_message(message.chat.id, mess,parse_mode='HTML')

        bot.register_next_step_handler(msg,more_questions,[program,None])
    else:
        call_delete_menu(message)
        mess = 'probably you did not choose option from buttons, so error occurred'
        bot.send_message(message.chat.id,mess)
        time.sleep(10)
        restart(message)

def insert_new_program_info(message):
    try:
        file_name = message.document.file_name
        file_info = bot.get_file(message.document.file_id)
        file_data = bot.download_file(file_info.file_path)
        print(type(file_data))
            
        if(file_name in mongodb_read.show_program_name('itmo_data')):
            mess = 'File name duplicated , please change it and upload it again'
            mess += '\nBack to menu...'
            bot.send_message(message.chat.id,mess)
            time.sleep(3)
            admin_start(message)
        else:
            table_name = file_name.split('.')[0]
            mycol = mongodb_read.mongodb_atlas('itmo_data',table_name)
            # if exists 
            mycol.drop()

            with open(file_name, 'wb') as new_file:
                new_file.write(file_data)

            with open(file_name, 'rb') as new_file:
                file_content = json.load(new_file)
            print(type(file_content))
            index = -1
            for i in file_content:
                index += 1
                for j in i['patterns']:
                    for k in i['responses']:
                        mycol.insert_one({
                            'tag':index,
                            'patterns':j,
                            'responses':k,
                            'time':mongodb_read.datetime.now(),
                            'access':False
                        })  
            #mycol.insert_many(file_content)  
            mess = f'new df is updated' 
            os.remove(file_name)
            bot.send_message(message.chat.id, mess)

            #retrain start
            t1=threading.Thread(target=retrain.mid_term_retrain)
            t1.start()
            mess = 'retrain starts'
            bot.send_message(message.chat.id,mess)
            
            time.sleep(10)
            admin_start(message)
    except:
        mess = """Probably json file has wrong format\nMake sure format follow like below\n
        [
        {},
        {}
        ]"""
        os.remove(file_name)
        bot.send_message(message.chat.id, mess)
        time.sleep(10)
        restart(message)
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
    markup.add(types.InlineKeyboardButton('Correct unknown questions', callback_data='correct_unknown_questions')) 
    markup.add(types.InlineKeyboardButton('Update responses', callback_data='Update_responses')) 
    markup.add(types.InlineKeyboardButton('Change password', callback_data='change_password'))    
    markup.add(types.InlineKeyboardButton('Statistics', callback_data='statistics_plot'))  
    markup.add(types.InlineKeyboardButton('add new program', callback_data='add new program'))     
    return markup


def admin_start(message):
    global msg_id
    mess = f'hi, <b>{message.from_user.first_name}</b>!\nWelcome to admin mode'
    #record name and id   
    markup = admin_menu()
    msg_id = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html').message_id
    #bot.register_next_step_handler(msg, force_button_click)

def update_unknown_datasets(message):

    mycol = mongodb_read.mongodb_atlas('global','unknown_response') 
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
        print('update_unknown_datasets')
        print(one_unknown)

        if(msg != None):
            indexs,_ = model_process(model,tokenizer,msg,max_length)
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)

            for i in range(len(indexs)):
                mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning') 
                found = mycol.find_one({'tag':indexs[i]}) 
                tag = str(found['tag'])
                response = found['responses']
                markup.add(f'{tag} {response}') 

            markup.add('New Topic')
            markup.add('Go to Trash')
            markup.add('Back to menu')

            mess = f"""Please choose most correspondent answer to question : \n<b>{msg}</b> about {one_unknown['program'].tolist()[0]}""" 
            mess += f'\nThey are order by most possible responses' 
            msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')

            #send rest of unknown to next func
            bot.register_next_step_handler(msg, insert_delete_unknow,one_unknown)
        else:
            mycol.delete_one({"_id":one_unknown['_id'].tolist()[0]}) 
            update_unknown_datasets(message)
def insert_delete_unknow(message,one_unknown):
    msg = message.text
    
    try:
        print(one_unknown)
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
            mycol = mongodb_read.mongodb_atlas('global','unknown_response')
            print('one_unknown is')
            print(one_unknown)
            print(one_unknown.to_dict('records')[0])

            mycol.delete_one({"_id":one_unknown['_id'].tolist()[0]}) 

            mycol = mongodb_read.mongodb_atlas('global','trash')
            one_unknown.pop('_id')
            mycol.insert_one(one_unknown.to_dict('records')[0]) 
            time.sleep(3)
            update_unknown_datasets(message)
        else:
            tag = int(msg.split(' ')[0])
            print('before deleted finished in unknown')
            print(one_unknown['_id'])
            print(tag)
            mycol = mongodb_read.mongodb_atlas('global','unknown_response')
    
            mycol.delete_one({"_id":one_unknown['_id'].tolist()[0]}) 
            print('deleted finished in unknown')

            mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')
            found = mycol.find_one({'tag':tag})
            tag = found['tag']
            print(tag)
            one_unknown['response'] = tag
            print('after tag')
            print('one_unknown is')
            print(one_unknown)
            one_unknown.pop('_id')
            print('one_unknown is')
            print(one_unknown)



            mycol = mongodb_read.mongodb_atlas('global','new_response')
            mycol.insert_one(one_unknown.to_dict('records')[0]) 
            print('insert finished in new')
            mycol = mongodb_read.mongodb_atlas('global','response_to_user')
            mycol.insert_one(one_unknown.to_dict('records')[0]) 
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
def update_responses(message):
    try:
        tag = int(message.text.split(' ')[0])
        print(tag)
        mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')
        found = mycol.find_one({'tag':tag})
        former_text = found['responses']
        mess = 'original text:\n'
        mess += '{}\n\n'.format(former_text)
        mess += 'template for question: \n'
        mess += '{}'.format(found['patterns'])
        bot.send_message(message.chat.id,mess)

        mess = f'Please write new response'
        text = bot.send_message(message.chat.id,mess)
        tag_former_text = []
        tag_former_text.append(tag)
        tag_former_text.append(former_text)
        bot.register_next_step_handler(text, confirm_options,tag_former_text)
    except:
        mess = 'probably you did not choose option from buttons, so error occurred'
        bot.send_message(message.chat.id,mess)
        time.sleep(3)
        admin_start(message)

def confirm_options(message,tag_former_text):
    try:
        text = message.text
        tag = tag_former_text[0]
        former_text = tag_former_text[1]
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        markup.add('Yes') 
        markup.add('No') 
        markup.add('Another response') 
        mess = f'Agree to change ?\n' 
        mess += 'From \n{}\n\n'.format(former_text) 
        mess += 'To :\n{}\n'.format(text) 

        msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
        tag_text = []
        tag_text.append(tag)
        tag_text.append(text)
        tag_text = []
        tag_text.append(tag)
        tag_text.append(text)
        bot.register_next_step_handler(msg, confirm_update,tag_text)
    except:
        mess = 'probably you did not choose option from buttons, so error occurred'
        bot.send_message(message.chat.id,mess)
        time.sleep(3)
        admin_start(message)
    
def confirm_update(message,tag_text):
    try:
        ans = message.text
        tag = tag_text[0]
        text = tag_text[1]
        mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')

        if(ans == 'Yes'):
            newvalues = { "$set": { "responses": text } }
            mycol.update_many({'tag':tag},newvalues)
            mess = f'successfully updated , back to menu'
            bot.send_message(message.chat.id,mess)
            time.sleep(3)
            admin_start(message)
        elif(ans == 'No'):
            mess = f'Please write again'
            bot.send_message(message.chat.id,mess)
            update_responses(tag)
        elif(ans == 'Another response'):
            mycol = mongodb_read.mongodb_atlas('itmo_data','big data and machine learning')
            x = mycol.find().sort('tag')
            train_data = pd.DataFrame(list(x))[['tag','responses']].drop_duplicates()
            tags = train_data.tag.tolist()
            responses = train_data.responses.tolist()
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
            for i in range(len(tags)):
                markup.add(f'{tags[i]} {responses[i]}') 

            mess = f'Please choose text which you want to correct' 
            msg = bot.send_message(message.chat.id, mess, reply_markup=markup, parse_mode='html')
            tag = msg.split(' ')[0]
            bot.register_next_step_handler(tag, update_responses)
        else:
            mess = 'Probably you did not choose option from buttons, so error occurred'
            bot.send_message(message.chat.id,mess)
            time.sleep(3)
            admin_start(message)
    except:
        mess = 'Error occurred, sorry for inconvenience\nRedirecting to menu'
        bot.send_message(message.chat.id,mess)
        time.sleep(3)
        admin_start(message)


def insert_new_topic(message,one_unknown):
    print('hi')
    msg = message.text
    mycol = mongodb_read.mongodb_atlas('global','unknown_response')

    mycol.delete_one({"_id":one_unknown['_id'].tolist()[0]}) 
    print('1')
    program = one_unknown['program']
    print('2')
    mycol = mongodb_read.mongodb_atlas('itmo_data',program[0])
    print('3')
    x= list(mycol.find().sort({ "tag" : -1 }).limit(1))
    label_num = x[0]['tag'] + 1
    access = x[0]['access']
    # add modify_index to new_response, it will change when model training is over
    mess = 'It may takes some time'
    bot.send_message(message.chat.id,mess)

    dict_example = {'tag':label_num,'patterns':one_unknown['message'].tolist()[0],'responses':msg,'time':mongodb_read.datetime.now()
                    ,'access':access}
    print(dict_example)
    mycol.insert_one(dict_example) 

    one_unknown.pop('_id')
    one_unknown['response'] = label_num
    mycol = mongodb_read.mongodb_atlas('global','new_response')
    mycol.insert_one(one_unknown.to_dict('records')[0]) 
        
    mycol = mongodb_read.mongodb_atlas('global','response_to_user')
    mycol.insert_one(one_unknown.to_dict('records')[0]) 

    mess = '1 new topic successfully updated'
    bot.send_message(message.chat.id,mess)

    #retrain start
    t1=threading.Thread(target=retrain.mid_term_retrain)
    t1.start()
    mess = 'retrain starts'
    bot.send_message(message.chat.id,mess)

    #back to check more unknown datasets
    update_unknown_datasets(message)

def change_password(message):
    new_password = message.text

    with open("password.txt", "r") as f:
        lines = f.readlines()
        lines[0] = new_password
        
        with open("password.txt",'w') as txt:
            txt.writelines(lines)
            
    mess = 'Please save password safely'
    bot.send_message(message.chat.id,mess)
    time.sleep(2)
    admin_start()
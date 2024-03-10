import pymongo
import telebot

token = '6705181314:AAH1F4h1C_rpM5pkcu3tXdeHkznDxIESz3o'
token2 = 'mongodb+srv://mongo:mongo@cluster0.gcj8po2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

bot = telebot.TeleBot(token, parse_mode='None')

        
myclient = pymongo.MongoClient(token2)
import pymongo
import pytz
from datetime import datetime

eastern_tz = pytz.timezone('Europe/Moscow')

password = 'ITMO'
token = '6705181314:AAH1F4h1C_rpM5pkcu3tXdeHkznDxIESz3o'
token2 = 'mongodb+srv://mongo:mongo@cluster0.gcj8po2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

        
myclient = pymongo.MongoClient(token2)

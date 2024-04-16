
from main import *
import tokens
import pymongo

from apscheduler.schedulers.background import BackgroundScheduler

from retrain import train_process
from recommend import train_recommend


myclient = pymongo.MongoClient(tokens.token2)

if __name__ == "__main__":
    bot.polling(none_stop=True)
    # while True:
    #     try:
    #         scheduler1 = BackgroundScheduler()
    #         scheduler1.add_job(id='Scheduled task', func=train_process, trigger='interval', weeks=4)
    #         scheduler1.add_job(id='Scheduled task1', func=train_recommend, trigger='interval', weeks=4)

    #         scheduler1.start()
        
    #         bot.polling(none_stop=True)
    #     except Exception as e:
    #         print(e) # or log it in some way
    #     print("Restarting main")        


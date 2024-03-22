
from main import *
import tokens
import pymongo
myclient = pymongo.MongoClient(tokens.token2)

if __name__ == "__main__":
    bot.polling(none_stop=True)

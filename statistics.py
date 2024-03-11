import pandas as pd
import matplotlib 
matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from main import *
from mongodb import *

@bot.message_handler(commands=['statistics'])
def show_statistics(message):
    # 
    mycol = mongodb_atlas('new_response')
    x = mycol.find()    
    df = pd.DataFrame(list(x))
    
    mycol = mongodb_atlas('new_response')
    x = mycol.find()    
    df_new = pd.DataFrame(list(x))
    df_new  = df_new.dropna(subset = ['message'])

    mycol = mongodb_atlas('training_data')
    x = mycol.find()    
    df_train = pd.DataFrame(list(x))
    df['accuracy'] = np.where(df['predicted'] == df['response'],1,0)

    
    mess = ''
    mess += f"{len(df.id.unique())} students have used this system"
    
    mess += "\n"
    accuracy = len(df[df['accuracy']==1])/len(df)
    mess += f"current accuracy is {accuracy}"

    #auto retrain
    if((accuracy < 0.7) & (len(df_new)>30)):
        pass
        retrain()
        
    # visitors plot by week
    df['accuracy'] = df['accuracy'].apply(lambda x : None if x==0 else x)
    grouped_data = df.groupby(pd.Grouper(key='time', freq='W'))
    x = grouped_data.count()
    x = x.reset_index()
    x.time = x.time.apply(lambda x : f'''{str(x).split('-')[1]}-{str(x).split('-')[2]}''')
    x.time = x.time.apply(lambda x : f'''{str(x).split(' ')[0]}''')
    
    
    plt.bar(x.time.tolist(), x.message.tolist(), edgecolor='purple', color='None')
    plt.plot(x.time.tolist(), x.accuracy.tolist())
    plt.savefig("visitors.png") 
    plt.show()
    
    bot.send_photo(message.chat.id, photo=open('visitors.png', 'rb'))
    
    # ranking for questions
    indexs = df.response.value_counts().index.tolist()
    counts = df.response.value_counts().tolist()
    mess += "\n"
    mess += "\n"
    mess +='most asked questions are'
    mess += "\n"
    for i,c in zip(indexs[:5],counts[:5]):
        mess +=f"{i} : {c}"
        mess += "\n"
    bot.send_message(message.chat.id, mess)

    plt.bar(indexs,counts)
    
      
    # saving the file.Make sure you  
    # use savefig() before show(). 
    plt.savefig("topics.png") 
    plt.show()
      
    bot.send_photo(message.chat.id, photo=open('topics.png', 'rb'))
    time.sleep(3)
    restart(message)

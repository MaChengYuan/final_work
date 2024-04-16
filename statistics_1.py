import pandas as pd
import matplotlib.pyplot as plt
import main 
import mongodb_read 
import numpy as np
from datetime import datetime,timedelta

def show_statistics(message):
    plt.style.use('bmh')
    end = datetime.now()
    start = end - timedelta(days = 365)

    mycol = mongodb_read.mongodb_atlas('global','new_response')
    x = mycol.find({"time":{"$gte":start}})      
    df = pd.DataFrame(list(x))
    
    mycol = mongodb_read.mongodb_atlas('global','new_response')
    x = mycol.find({"time":{"$gte":start}})      
    df_new = pd.DataFrame(list(x))
    df_new  = df_new.dropna(subset = ['message'])

    df['accuracy'] = np.where(df['predicted'] == df['response'],1,0)

  
    # combine known and unknown datasets
    acc_df = df[df.predicted == df.response].reset_index(drop=True)
    
    mycol = mongodb_read.mongodb_atlas('global','unknown_response')
    x = mycol.find({"time":{"$gte":start}})    
    un_df = pd.DataFrame(list(x))
    
    alldf = pd.concat([acc_df,un_df]).reset_index(drop=True)

    mess = ''
    mess += f"{len(alldf.id.unique())} students have used this system"
    
    mess += "\n"
    accuracy = len(alldf[alldf['accuracy']==1])/len(alldf)
    
    mess += f"current accuracy rate is {accuracy}"

    #auto retrain
    if((accuracy < 0.7) & (len(df_new)>30)):
        pass
        
    alldf['acc'] = np.where(alldf['predicted'] == alldf['response'],1,0)    
    alldf['inacc'] = np.where(alldf.predicted != alldf.response,1,0)
    alldf = alldf[['time','acc','inacc']]
    x = alldf.groupby(pd.Grouper(key='time', axis = 0,freq='W')).sum().reset_index().sort_values('time',ascending = False)

    # visitors plot by week
    x.time = x.time.apply(lambda x : f'''{str(x).split('-')[1]}-{str(x).split('-')[2]}''')
    x.time = x.time.apply(lambda x : f'''{str(x).split(' ')[0]}''')

    # only print last 10 weeks
    X_axis = np.arange(len(x)) 
    width = 0.2
    plt.barh(X_axis - width/2, x.acc.tolist()[:10], width, color='red', label='accuracy')
    plt.barh(X_axis + width/2, x.inacc.tolist()[:10], width, color='green', label='inaccuracy')
    plt.yticks(X_axis, x.time.tolist()) 
    
    def gen_label(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.legend()
    plt.savefig("visitors.png")
    plt.show()
    
    # new image just in case
    plt.figure() 
    
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

    rects = plt.bar(indexs,counts)
    
    gen_label(rects)  
    # saving the file.Make sure you  
    # use savefig() before show(). 
    plt.savefig("topics.png") 
    plt.show()
      
    main.bot.send_message(message.chat.id, mess)
    main.bot.send_photo(message.chat.id, photo=open('visitors.png', 'rb'))  
    main.bot.send_photo(message.chat.id, photo=open('topics.png', 'rb'))


import pandas as pd
import matplotlib.pyplot as plt
from main import *
from mongodb import *


#stastical plot
@bot.message_handler(commands=['statistics'])
def show_statistics(message):
    mycol = mongodb_atlas('new_response')
    x = mycol.find()    
    df = pd.DataFrame(list(x))
    df['accuracy'] = np.where(df['predicted'] == df['response'],1,0)
    mess = ''
    mess += f"{len(df.id.unique())} students have tried to get information"
    mess += "\n"
    mess += f"accuracy of model so far {len(df[df['accuracy']==1])/len(df)}"

    # ranking for questions
    indexs = df.response.value_counts().index.tolist()
    counts = df.response.value_counts().tolist()
    mess += "\n"
    mess += "\n"
    mess +='most asked questions are'
    mess += "\n"
    for i,c in zip(indexs,counts):
        mess +=f"{i} : {c}"
        mess += "\n"
    bot.send_message(message.chat.id, mess)
 
  
    # creating plotting data 
    xaxis =[1, 4, 9, 16, 25, 36, 49, 64, 81, 100] 
    yaxis =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
      
    # plotting  
    plt.plot(xaxis, yaxis) 
    plt.xlabel("X") 
    plt.ylabel("Y") 
      
    # saving the file.Make sure you  
    # use savefig() before show(). 
    plt.savefig("squares.png") 
      
    bot.send_photo(message.chat.id, photo=open('squares.png', 'rb'))
    time.sleep(3)
    restart(message)

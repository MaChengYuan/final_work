#mongodb
import mongodb_read 
#model
import NLPmodel 

# import libraries
import random
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import torch
from transformers import RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import warnings
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
nltk.download('stopwords')
warnings.filterwarnings("ignore", category=UserWarning)

from eda import gen_eda
from cPosGpt2 import train_cposgpt2_and_augment

def set_seed(seed_value):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)


scl_model_path = r"itmo_model.pt"
cross_model_path = r"itmo_model.pt"
stop_words = set(stopwords.words('english'))

# a function for preprocessing text
def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Load the BERT tokenizer


# Create a function to tokenize a set of texts
def preprocessing_for_bert(tokenizer,data, MAX_LEN):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def prepare_data(tokenizer,train_ds,val_ds=None,aug_path = None,sample_num = 10 , seed = 32, all=True , aug = False,aug_num = 6):
    # load data
    # for tsv
    # there is no valid_path
      
    global num_classes
    num_classes = len(train_ds['label'].unique())
    
    #original one
    if(all == False):
        train_df = [train_ds.loc[train_ds.label == i].sample(n=sample, random_state=seed) for i in
                    train_ds.label.unique()]
        train_df = pd.concat(train_df, axis=0).sample(frac=1)
    else :
        #for all
        train_df = train_ds
    
    train_df = train_df[['sentence','label']]
    #print(train_df.head(3))
    # data augmentation 
    if(aug == True):
        #indexs = train_df.index.values.tolist()
        aug_df =  pd.read_csv(aug_path, sep='\t')
        #aug_df = [aug_ds[i*aug_num:i*aug_num+aug_num] for i in indexs]
        
        #print(aug_df[:aug_num*2])
        aug_df = pd.concat(aug_df, axis=0).sample(frac=1)
        train_df = pd.concat([train_df,aug_df], axis=0).sample(frac=1).reset_index(drop=True)
    
    if(val_ds == None):
        sample = 5
        
        val_df = [train_df.loc[train_df.label == i].sample(n=sample,replace = True, random_state=seed) for i in
                    train_df.label.unique()]
        val_df = pd.concat(val_df, axis=0).sample(frac=1).reset_index(drop=True)

    
    train_text = train_df["sentence"].tolist()
    train_label = train_df["label"].tolist()
    val_text = val_df["sentence"].tolist()
    val_label = val_df["label"].tolist()
    

    # Concatenate train data and test data
    all_text = np.concatenate([train_text, val_text], axis=0)

    # Encode our concatenated data
    encoded_text = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_text]
    global MAX_LEN
    # Find the maximum length
    MAX_LEN = max([len(sent) for sent in encoded_text])

    # preprocessing train data
    for i in range(len(train_text)):
        train_text[i] = text_preprocessing(train_text[i])

    # preprocessing validation data
    for i in range(len(val_text)):
        val_text[i] = text_preprocessing(val_text[i])

    # Run function `preprocessing_for_bert` on the train set and the validation set
    # print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(tokenizer,train_text, MAX_LEN)
    val_inputs, val_masks = preprocessing_for_bert(tokenizer,val_text, MAX_LEN)


    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(train_label)
    val_labels = torch.tensor(val_label)


    # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
    batch_size = 16

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


    return train_dataloader, val_dataloader


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    


def initialize_model(model,hidden = 16 , num_labels = 2 ,feature_remove_max=True):
    """Initialize the Classifier, the optimizer and the learning rate scheduler.
    """

    # Instantiate Bert Classifier
    if(model == NLPmodel.QModel_Classifier):
        model_classifier = model(1024, hidden_dim=hidden, num_labels = num_labels, dropout=0.1,feature_remove_max=feature_remove_max)
    else:
        model_classifier = model(1024, hidden_dim=hidden, num_labels = num_labels, dropout=0.1)

    # Tell PyTorch to run the model on GPU
    model_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(model_classifier.parameters(),
                      lr=4e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )

    return model_classifier, optimizer


def contrastive_loss(temp, embedding, label):
    """calculate the contrastive loss
    """
    nsamples, nx, ny = embedding.shape
    embedding = embedding.reshape((nsamples,nx*ny))
    
    # cosine similarity between embeddings
    cosine_sim = cosine_similarity(embedding, embedding)
    # remove diagonal elements from matrix
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    # apply temprature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    # calculate row sum
    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))
    # calculate outer sum
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        inner_sum = 0
        # calculate inner sum
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss


def evaluate(model, val_dataloader, tem, lam, scl):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    loss_fn = nn.CrossEntropyLoss()
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits, h_s,_ = model(b_input_ids, b_attn_mask)

        # Compute loss
        if scl:
            cross_loss = loss_fn(logits, b_labels)
            contrastive_l = contrastive_loss(tem, h_s.cpu().detach().numpy(), b_labels)
            loss = (lam * contrastive_l) + (1 - lam) * (cross_loss)
            val_loss.append(loss.item())
        else:
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            #self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def train(model,optimizer,train_dataloader, tem, lam, scl,epoch = 40,val_dataloader=None, evaluation=False,patience = 25):
    """Train the BertClassifier model.
    """
    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()
    # Start training loop
    print("Start training...\n")
    val_list = []
    train_list = []
    best_validation_loss = float('inf')
    early_stopper = EarlyStopper(patience=patience, min_delta=0)

    for e in range(epoch):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        e = e + 1
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Train Accuracy':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 86)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        train_accuracy = []
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits, hiden_state,_ = model(b_input_ids, b_attn_mask)

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            train_accuracy.append(accuracy)

            # Compute loss
            if scl == True:
                cross_loss = loss_fn(logits, b_labels)
                contrastive_l = contrastive_loss(tem, hiden_state.cpu().detach().numpy(), b_labels)
                loss = (lam * contrastive_l) + (1 - lam) * (cross_loss)
            if scl == False:
                loss = loss_fn(logits, b_labels)

            batch_loss += loss.item()
            total_loss += loss.item()
            print(
                f"{e:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {accuracy:^14.6} | {'-':^10} | {'-':^9} | {'-':^9.2}")

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()

        # Reset batch tracking variables
        batch_loss, batch_counts = 0, 0
        t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        total_accuracy = np.mean(train_accuracy)
        train_list.append(avg_train_loss)

        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader, tem, lam, scl)
            val_list.append(val_loss)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print("-" * 86)
            print(
                f"{'end':^7} | {'-':^7} | {avg_train_loss:^12.6f} | {total_accuracy:^14.6} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 86)
        print("\n")

    
        
        if (val_loss < best_validation_loss) and scl == True:
            best_validation_loss = val_loss
            torch.save(model.state_dict(), scl_model_path)
        elif (val_loss < best_validation_loss) and scl == False:
            best_validation_loss = val_loss
            torch.save(model.state_dict(), cross_model_path)


        #early stopping
        #print(early_stopper.counter)
        if early_stopper.early_stop(val_loss):  
            break


    # plot train and valid loss
    plt.plot(list(range(len(val_list))), val_list, label="validation loss")
    plt.plot(list(range(len(train_list))), train_list, label="training loss")
    plt.title('loss')
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    print("Training complete!")

    return best_validation_loss, val_accuracy


from tqdm import tqdm
def test_evaluate(model,model_path, test_dataloader,hidden=16,num_labels=2,feature_remove_max=False):
    """After the completion of each training epoch, measure the model's performance
    on our vtest set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    if(model == NLPmodel.QModel_Classifier):
        model = model(1024,hidden, num_labels=num_classes, dropout=0.1,feature_remove_max=feature_remove_max)
    else:
        model = model(1024,hidden, num_labels=num_classes, dropout=0.1)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Tracking variables
    test_accuracy = []
    predict = []
    y_true = []

    # For each batch in our test set...
    for batch in tqdm(test_dataloader):
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits, _ ,_= model(b_input_ids, b_attn_mask)

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        predict += preds.tolist()
        y_true += b_labels.tolist()

    # Accuracy
    print(f'Accuracy: {accuracy_score(y_true, predict)}')

    # Recall
    print(f'Recall: {recall_score(y_true, predict, average=None)}')

    # Precision
    print(f'Precision: {precision_score(y_true, predict, average=None)}')

    # F1_score
    print(f'F1_score: {f1_score(y_true, predict, average=None)}')

    return accuracy_score(y_true, predict)
    
#test_evaluate(cross_model_path, test_dataloader)
#test_evaluate(scl_model_path, test_dataloader)

# scl_test_acc = test_evaluate(scl_model_path, test_dataloader)

def model_predict(model,hidden, model_path, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model = model(1024,hidden, num_classes, dropout=0.1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits, _,_ = model(b_input_ids, b_attn_mask)
        preds = torch.argmax(logits, dim=1).flatten()
        all_logits += preds.tolist()

    # Concatenate logits from each batch
    # all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    # probs = F.softmax(all_logits, dim=1).cpu().numpy()
    # predict = np.argmax(probs)

    return all_logits

# delete defect column : time is None , label = string , message from users has not insufficient information
def clean_db(df):
    #drop time = null
    time_na = df[df['time'].isna()]
    df = df.dropna(subset=['time']).reset_index(drop=True)
    for i in time_na._id:
        try:
            mycol = mongodb_read.mongodb_atlas('new_response')
            mycol.delete_one({'_id':i})
        except:
            print('something wrong happen')
            pass

    #drop label = str
    label_non_str = df[df['response'].apply(lambda x: isinstance(x, str))]
    df = df[~df['response'].apply(lambda x: isinstance(x, str))]
    
    for i in label_non_str._id:
        try:
            mycol = mongodb_read.mongodb_atlas('new_response')
            mycol.delete_one({'_id':i})
        except:
            print('something wrong happen')
            pass

    # remove datasets generated from recommendation
    df = df.dropna(subset = ['message']).reset_index(drop=True)

    # locate data = None afte clean
    df['clean'] = df.message.apply(lambda x: clean_punctuation(x))
    df['clean'] = df.clean.apply(lambda x: wordpunct_tokenize(x))
    df['clean'] = df.clean.apply(lambda x: [w for w in x if not w.lower() in stop_words])
    defect_message = df[df['clean'].apply(lambda x: len(x)==0)]
    for i in defect_message._id:
        try:
            mycol.delete_one({'_id':i})
        except:
            print('something wrong happen')
            pass
        
    df = df[df['clean'].apply(lambda x: len(x)!=0)]
       
    print(f"Datasets are cleansed")
    return df

def clean_punctuation(string):
        try:
            res = re.sub(r'[^\w\s]', '', string)
            return res
        except:
            return string


def train_process():
    MAX_LEN = 64
    epoch =40 # number of epochs

    scl = True  # if True -> scl + cross entropy loss. else just cross entropy loss
    temprature = 0.3  # temprature for contrastive loss
    lam = 0.9  # lambda for loss
    patience=12 # early stop
    hidden = 512
    seed = 49    # seed

    # for eda
    alpha = 0.1
    num_aug = 6
    # for pos_gpt2
    second_num_aug = 4

    # original datasets plus new_responses 
    mycol = mongodb_read.mongodb_atlas('new_response')
    x = mycol.find()    
    df = pd.DataFrame(list(x))
    # clean and remove unneccesary training datas
    df = clean_db(df)
    df = df[['response','message']]
    new = df.rename(columns={'response':'label','message':'sentence'})

    mycol = mongodb_read.mongodb_atlas('training_data')
    x = mycol.find()    
    df = pd.DataFrame(list(x))
    df = df.rename(columns={'tag':'label','patterns':'sentence'})
    traindf = df[['label','sentence']]

    df = pd.concat([traindf, new], axis=0).sample(frac=1).reset_index(drop=True)

    #dropna in case 
    df = df.dropna(subset = ['sentence']).reset_index(drop=True)


    
    df = df[['label','sentence']]


    #main bot augmentation file
    output_file = 'main_bot'
    os.makedirs(output_file, exist_ok=True)
               
    #eda
    print('eda phase')
    
    file_name = 'eda.tsv'
    output_dir = os.path.join(output_file, file_name)
    gen_eda(df,output_dir , alpha=alpha, num_aug=num_aug , reverse = False)
    
    #from transformers import GPTNeoForCausalLM
    from transformers import GPT2Tokenizer
    from transformers import GPT2LMHeadModel
    
    #gpt2
    print('GPT phase')
    
    
    GPT2_MODEL = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL,
                                            cache_dir='transformers_cache')
    
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL,
                                                  do_lower_case=True,
                                              cache_dir='transformers_cache')
    
    #GPT2_MODEL = 'EleutherAI/gpt-neo-1.3B' 
    #model = GPTNeoForCausalLM.from_pretrained(GPT2_MODEL,cache_dir='transformers_cache')

    #eda + gpt2     

    file_name = 'posgpt2_eda.tsv'    
    x = f'{output_file}/eda.tsv'
    train_df = pd.read_csv(x,sep='\t')

    sample = 3    
    val_df = [train_df.loc[train_df.label == i].sample(n=sample, random_state=seed) for i in
                    train_df.label.unique()]
    val_df = pd.concat(val_df, axis=0).sample(frac=1)

    
    train_cposgpt2_and_augment(model,tokenizer,train_df,val_df,output=output_file,file_name=file_name,seed = 1234,max_seq_length = MAX_LEN,sample_num=second_num_aug,num_train_epochs=5)

    aug_path = f'{output_file}/posgpt2_eda.tsv'

    set_seed(seed)
    print('pre-process phase')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True) 
    train_dataloader, val_dataloader = prepare_data(tokenizer,df,None,aug_path, sample_num=10\
                                                                         , seed = seed , all=True,aug=True,aug_num = num_aug*second_num_aug)
    
    # without contrasive loss
    
    bert_classifier, optimizer = initialize_model(NLPmodel.QModel_Classifier,hidden=hidden,num_labels = num_classes)
    scl = False

    print('training phase')
    val_loss, val_accuracy = train(bert_classifier,optimizer, train_dataloader, temprature, lam, scl, epoch ,val_dataloader, evaluation=True,patience=patience)

    #test 
    #test_accuracy = test_evaluate(QModel_Classifier,cross_model_path, test_dataloader,hidden=hidden,num_labels=num_classes,feature_remove_max=True)

    
if __name__ == "__main__":

        train_process()
        os.rmdir("main_bot")
    

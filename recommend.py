import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
#from metrics import compute_metrics
from datasets import (CausalLMDataset, CausalLMPredictionDataset, MaskedLMDataset,
                          MaskedLMPredictionDataset, PaddingCollateFn)
from models import RNN, BERT4Rec, SASRec
from modules import SeqRec, SeqRecWithSampling
from preprocess import add_time_idx
import pickle
from tqdm import tqdm
import mongodb_read

class Args():
    def __init__(self):
        super(Args, self).__init__()
        self.cuda_visible_devices= 0
        self.data_path= 'Beauty.txt'
        #data_path= 'ITMO.txt'
        
        self.full_negative_sampling= False
        self.num_negatives = None

        self.batch_size= 128
        self.test_batch_size= 256
        self.num_workers= 8
        self.validation_size= 10000
        self.model= "SASRec"
        self.maxlen= 200
        self.hidden_units= 64
        self.num_blocks= 2
        self.num_heads= 1
        self.dropout_rate= 0.1

        self.lr= 0.001
        self.predict_top_k= 10
        self.filter_seen= True

        self.max_epochs= 100
        self.patience= 10
        self.sampled_metrics= False
        self.top_k_metrics=10
        self.add_head = False

        mycol = mongodb_read.mongodb_atlas('training_data')
        x = mycol.find() 
        df = pd.DataFrame(list(x))
        self.item_id_max = len(df.tag.value_counts())
        self.max_length = self.item_id_max 

def train(seqrec_module, train_loader,config,eval_loader = None):
    
    optimizer = AdamW(seqrec_module.model.parameters(),
                          lr=4e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )
    train_list = []
    val_list = []
    model_path ='recommendation.pth'
    best_ndcg = -float('inf')
    update = 0
    
    for e in tqdm(range(config.max_epochs)):
        e = e + 1
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'ndcg':^12} | {'hit_rate':^10} | {'mrr':^9} | {'Elapsed':^9}")
        print("-" * 86)
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss = 0
        batch_counts = 0
        seqrec_module.model.train()
        
        for step, batch in enumerate(train_loader):
            batch_counts += 1
            
            #print(batch['input_ids'][0][0].item())       
            #print(type(batch['input_ids'][0][0].item()))
            #if(batch['input_ids'][0][0].item() == 0.0):
            #    print(batch['input_ids'][0])
            seqrec_module.model.zero_grad()
            
            loss = seqrec_module.training_step(batch,step)
            total_loss += loss
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(seqrec_module.model.parameters(), 1.0)
    
            # Update parameters and the learning rate
            optimizer.step()
            
            print(
                    f"{e:^7} | {step:^7} | {total_loss / batch_counts:^12.6f} | {'-':^14.6} | {'-':^10} | {'-':^9} | {'-':^9.2}")
    
        train_list.append(total_loss/len(train_loader))
        if eval_loader:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.        
            ndcg,hit_rate,mrr = 0, 0, 0
            matrix = []
            for step, batch in enumerate(eval_loader):
                x,y,z = seqrec_module.validation_step(batch,step)
                ndcg += x
                hit_rate += y
                mrr += z
    
            time_elapsed = time.time() - t0_epoch
    
            print("-" * 86)
            print(
                f"{'end':^7} | {'-':^7} | {total_loss/len(train_loader):^12.6f} | {ndcg:^14.6} | {hit_rate:^10.6f} | {mrr:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 86)
            print("\n")
    
            
            matrix.append(ndcg)
            matrix.append(hit_rate)
            matrix.append(mrr)
                
            val_list.append(matrix)

        
        if (ndcg > 0):
            update += 1
            print('model is updated')
            best_ndcg = ndcg
            torch.save(seqrec_module.model.state_dict(), model_path)

    # if no update at all , update the model in last epoch
    if(update == 0):
        torch.save(seqrec_module.model.state_dict(), model_path)
            
    print(f'training process is over, model is saved in path {model_path}')

def preprocess(config,itmodf=None):
    
    data = itmodf 
    #pd.read_csv(config.data_path, sep=' ', header=None, names=['user_id', 'item_id'])
    data = add_time_idx(data, sort=False)

    # index 1 is used for masking value
    if config.model == 'BERT4Rec':
        data.item_id += 1

    # split dataset 
    train = data[data.time_idx_reversed >= 2]
    validation = data[data.time_idx_reversed == 1]
    validation_full = data[data.time_idx_reversed >= 1]
    test = data[data.time_idx_reversed == 0]

    #dataloader
    validation_size = config.validation_size
    validation_users = validation_full.user_id.unique()
    if validation_size and (validation_size < len(validation_users)):
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
    
    if config.model in ['SASRec', 'RNN']:
        train_dataset = CausalLMDataset(train, config.max_length,config.full_negative_sampling)
        eval_dataset = CausalLMPredictionDataset(
            validation_full[validation_full.user_id.isin(validation_users)],
            max_length=config.max_length, validation_mode=True)
    elif config.model == 'BERT4Rec':
        train_dataset = MaskedLMDataset(train, config.max_length,config.full_negative_sampling)
        eval_dataset = MaskedLMPredictionDataset(
            validation_full[validation_full.user_id.isin(validation_users)],
            max_length=config.max_length, validation_mode=True)
    
    train_loader = DataLoader(
        train_dataset, shuffle=True,
        collate_fn=PaddingCollateFn(),
        batch_size=config.batch_size)
    eval_loader = DataLoader(
        eval_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=config.test_batch_size)

    # load model

    item_count = config.item_id_max 
    
    if hasattr(config, 'num_negatives') and config.num_negatives:
        config.add_head = False
    else:
        config.add_head = True
    
    if config.model == 'SASRec':
        model = SASRec(item_num=item_count, maxlen=config.max_length, add_head=config.add_head)
    if config.model == 'BERT4Rec':
        model = BERT4Rec(vocab_size=item_count + 1, maxlen=config.max_length, add_head=config.add_head,
                         bert_config=config.model_params)
    elif config.model == 'RNN':
        model = RNN(vocab_size=item_count + 1, add_head=config.add_head,
                    rnn_config=config.model_params)

    #load module
    if (config.num_negatives != None) :
        seqrec_module = SeqRecWithSampling(model, config.lr,config.predict_top_k,config.filter_seen)
    else:
        seqrec_module = SeqRec(model, config.lr,config.predict_top_k,config.filter_seen)

    # save config
    with open('recommend_config.pickle', 'wb') as f:
        pickle.dump(config, f)
    
    return seqrec_module , train_loader , eval_loader


def filter_seen_items(preds, scores, seen_items,config):
    
    seen_items = torch.as_tensor([lbl for lbl in seen_items])
    seen_items = seen_items.reshape([1, len(seen_items)])
    max_len = seen_items.size(1)
    
    scores = scores[:, :config.predict_top_k + max_len]
    preds = preds[:, :config.predict_top_k + max_len]

    final_preds, final_scores = [], []
    for i in range(preds.size(0)):
        not_seen_indexes = torch.isin(preds[i], seen_items[i], invert=True)
        pred = preds[i, not_seen_indexes][:config.predict_top_k]
        score = scores[i, not_seen_indexes][:config.predict_top_k]
        final_preds.append(pred)
        final_scores.append(score)

    final_preds = torch.vstack(final_preds)
    final_scores = torch.vstack(final_scores)

    return final_preds, final_scores
    

def predict(config,user_history_topic):
    with open(f'recommend_config.pickle', 'rb') as file2:
        s1_new = pickle.load(file2)

    item_count = config.item_id_max

    if hasattr(config, 'num_negatives') and config.num_negatives:
        config.add_head = False
    else:
        config.add_head = True
    
    if config.model == 'SASRec':
        model = SASRec(item_num=item_count, maxlen=config.max_length, add_head=config.add_head)
    if config.model == 'BERT4Rec':
        model = BERT4Rec(vocab_size=item_count + 1, maxlen=config.max_length, add_head=config.add_head,
                         bert_config=config.model_params)
    elif config.model == 'RNN':
        model = RNN(vocab_size=item_count + 1, add_head=config.add_head,
                    rnn_config=config.model_params)
        
    model_path ='recommendation.pth'
    model.load_state_dict(torch.load(model_path))

    # input preprocess
    s = user_history_topic
    #s = [1,2,3]
    s_mask = []
    while(len(s_mask) < config.max_length):
        if(len(s_mask)<len(s)):
            
            s_mask.append(1)    
        else:
            
            s_mask.append(0)
    
    while(len(s) < config.max_length):
        s.append(0)
        
    s = torch.as_tensor([lbl for lbl in s])
    s_mask = torch.as_tensor([lbl for lbl in s_mask])    
    s = s.reshape([1, len(s)])

    # input process
    outputs = model(s,s_mask)
    
    rows_ids = torch.arange(s.shape[0], dtype=torch.long)
    last_item_idx = (s !=0).sum(axis=1) - 1
    
    # for potential item
    preds = outputs[rows_ids, last_item_idx, :]
    
    scores, preds = torch.sort(preds, descending=True)
    
    #list of already viewed before
    seen_items = user_history_topic
    print(f'config.item_id_max : {config.item_id_max}')
    if(len(seen_items) != 0):
        preds,scores = filter_seen_items(preds, scores, seen_items,config)
    else:
        scores = scores[:, :config.predict_top_k]
        preds = preds[:, :config.predict_top_k]

    return preds , scores


def sequence_df(df,DA = False):
    # make history sequence based on user
    df2 = df.drop_duplicates(["id", "response"])
    x = df2.sort_values(['time']).groupby('id')['response'].apply(list).tolist()
    
    DA = DA
    if(DA ==  True):
        training_sets = []
        n = 3
        for y in x:
            z = [y[n:n+3] for n in range(len(y)-2)]
            for i in z :
                training_sets.append(i)
            
        training_sets
    else:
        training_sets = x
        
    # indexs for each sequence
    indexs = []
    for i in range(len(training_sets)):
        index = len(training_sets[i])*[i]
        indexs.append(index)

    df_indexs = []
    df_training_sets = []
    for i in range(len(training_sets)):
        df_indexs.extend(indexs[i])
        df_training_sets.extend(training_sets[i])
    
    sequnce_df = pd.DataFrame(
        {'user_id': df_indexs,
         'item_id': df_training_sets
        })
    
    return sequnce_df


if __name__ == "__main__":
    config = Args()
    
    mycol = mongodb_read.mongodb_atlas('new_response')
    x = mycol.find() 
    df = pd.DataFrame(list(x))

    df = sequence_df(df,DA=False)
    #print(df.head())
    df.item_id = df.item_id.apply(lambda x : x+1)
    #print(df.head())
    seqrec_module , train_loader , eval_loader = preprocess(config,df)
    print(config.item_id_max)
    train(seqrec_module, train_loader,config,eval_loader = eval_loader)

    # predict template
    #history = [0]
    #preds, scores = predict(config,history)
    #print(preds)



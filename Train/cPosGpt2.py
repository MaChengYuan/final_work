import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

from transformers import AdamW

import nltk
from copy import deepcopy
from nltk import word_tokenize
from nltk import StanfordTagger
import random 
import re
import math
from nltk.corpus import stopwords
 
nltk.download('stopwords')
stops = stopwords.words('english')
                        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT2_MODEL = 'gpt2'
EOS_TOKEN = '<|endoftext|>'
SEP_TOKEN = '<SEP>'

STOP_TOKENS = [EOS_TOKEN, '<']


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, examples):
        self.examples = examples


def convert_examples_to_features(examples,examples_labels, block_size, tokenizer, seed=12345):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    text = ""
    for (ex_index, example) in enumerate(zip(examples,examples_labels)):
        if(ex_index==0):
            continue
        
        if ex_index:

            text += " " + example[1] + SEP_TOKEN + example[0] + EOS_TOKEN
        else:
            text += example[1] + SEP_TOKEN  + example[0] + EOS_TOKEN

    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    for i in range(0, len(tokenized_text) - block_size + 1,
                   block_size):  # Truncate in block of block_size
        features.append(InputFeatures(
            examples=tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size])))

    return features


def prepare_data(features):
    all_input_ids = torch.tensor([f.examples for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.examples for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_labels)
    return tensor_data

def compute_dev_loss(model, dev_dataloader):
    model.eval()
    sum_loss = 0.
    for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'labels': batch[1]}

        outputs = model(**inputs)
        loss = outputs[0]
        sum_loss += loss.item()
    return sum_loss




def augment_train_data(model, tokenizer, train_examples,train_label, output_dir,file_name,prefix,max_seq_length,sample_num,temperature,top_k,top_p,repetition_penalty):
    # load the best moщdel
    best_model_path = os.path.join(output_dir, "best_PostagGpt2.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
    else:
        raise ValueError("Unable to find the saved model at {}".format(best_model_path))
    prefix_size = prefix
    save_train_path = os.path.join(output_dir, file_name)
    save_train_file = open(save_train_path, 'w')

    tsv_writer = csv.writer(save_train_file, delimiter='\t')
    tsv_writer.writerow(['label','sentence'])
    
    prefix_text = None
    for ex_index, example in enumerate(zip(train_examples,train_label)):

        model.eval()

        words = tokenizer.tokenize(example[0])
        tag_list,no_g_words = get_postags(words)
        
        prefix_size = 0
        
        index_noun = []
        for i in range(len(tag_list)):
            if(((tag_list[i] == 'NN') or (tag_list[i] == 'NNP')) and (no_g_words[i].lower() not in stops)):      
                index_noun.append(i)

        if(len(index_noun) == 0):
            for i in range(len(tag_list)):
                if(((tag_list[i][:2] == 'VB')) and (no_g_words[i].lower() not in stops)):      
                    index_noun.append(i)


        
        
        if(len(index_noun) == 0):
            
            valid_index = []
            for index in range(len(tag_list)):
                res = bool(re.match('[a-zA-Z\s]+$', tag_list[index])) 
                if(res == True):
                    valid_index.append(index)
                    
            prefix_size = random.choice(list(range(len(valid_index)))) + 1


            if prefix_size > 0:
                    prefix_text = " ".join(example[0].split(' ')[:prefix_size])
                    raw_text = example[1] + SEP_TOKEN + prefix_text
            else:
                raw_text = example[1] + SEP_TOKEN
    
            context_tokens = tokenizer.encode(raw_text, return_tensors='pt').to(device)
            
            out = model.generate(
                input_ids=context_tokens,
                max_length=max_seq_length,
                num_return_sequences=sample_num,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=50256
            )
    
            out = out[:, len(context_tokens):].tolist()

            for o in out:

                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                eosn_index = 128
                for stop_token in STOP_TOKENS:
                    idx = text.find(stop_token)
                    if idx > 0:
                        eosn_index = min(eosn_index, idx)
                text = text[: eosn_index]
                text = text.replace("\n", " ").replace(EOS_TOKEN, ' ').strip()
                if prefix_size > 0:
                    text = prefix_text + " " + text
                tsv_writer.writerow([example[1], text])
        else:
            # sample_num must be divible to index_noun
            
            if(sample_num == 1):
                index_noun = random.sample(index_noun,1)
            else:
                if((len(index_noun) > sample_num) or ((sample_num%len(index_noun))!=0)):
                    number_index_num = math.gcd(len(index_noun), sample_num)
                    index_noun = random.sample(index_noun,number_index_num)
                    #index_noun = index_noun[:3]
                

            for j in index_noun: 
          
                prefix_size = j+1
                
                if prefix_size > 0:
                    prefix_text = " ".join(example[0].split(' ')[:prefix_size])
                    raw_text = example[1] + SEP_TOKEN + prefix_text
                else:
                    raw_text = example[1] + SEP_TOKEN

                context_tokens = tokenizer.encode(raw_text, return_tensors='pt').to(device)
                

                out = model.generate(
                    input_ids=context_tokens,
                    max_length=max_seq_length,
                    num_return_sequences=int(sample_num / len(index_noun)),
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=50256
                )
        
                out = out[:, len(context_tokens):].tolist()

                for o in out:
                    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                    eosn_index = 128
                    for stop_token in STOP_TOKENS:
                        idx = text.find(stop_token)
                        if idx > 0:
                            eosn_index = min(eosn_index, idx)
                    text = text[: eosn_index]
                    text = text.replace("\n", " ").replace(EOS_TOKEN, ' ').replace(SEP_TOKEN, ' ').strip()
                    text = text[: eosn_index]
                    text = text.split('.')[0]
                    text = text.split('?')[0]
                    text = text.split('</SEP>')[0]

                    #if prefix_size > 0:
                    #    text = prefix_text + " " + text
                    tsv_writer.writerow([example[1], text])

def get_postags(row1):
    row = deepcopy(row1)
    
    for i in range(len(row)) : 

        row[i] = row[i].replace('Ġ','')

    
    postags = nltk.pos_tag(row)
    list_classes = list()
    for word in postags:
        list_classes.append(word[1])

    
    return list_classes, row


def train_cposgpt2_and_augment(model,tokenizer,train_df,val_df,output='PosGpt2_eda',file_name='augment.tsv',seed = 1234,max_seq_length = 64,sample_num=6,num_train_epochs=8):

    seed = seed
    num_train_epochs = num_train_epochs
    train_batch_size = 8 
    learning_rate = 4e-5
    max_seq_length = max_seq_length
    prefix,temperature,top_k,top_p,repetition_penalty = 3 , 1 , 35 , 1 , 1
    block_size = 64
    output_dir = output
    sample_ratio , temp , sample_num = 7 , 1 , sample_num


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output, exist_ok=True)

    # load train and dev data
   
    train_df['label'] = train_df['label'].apply(lambda x:str(x))
    train_examples = train_df['sentence']
    
    train_label = train_df['label']


    block_size = min(block_size, tokenizer.max_len_single_sentence)

    model.to(device)

    # train data
    train_features = convert_examples_to_features(train_examples,train_label,
                                                  block_size,
                                                  tokenizer, seed)
    train_data = prepare_data(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=train_batch_size)
    

    
    val_df['label'] = val_df['label'].apply(lambda x:str(x))
    dev_examples = val_df['sentence']
    
    dev_label = val_df['label']
    
    dev_features = convert_examples_to_features(dev_examples,dev_label,
                                                block_size,
                                                tokenizer, seed)
    dev_data = prepare_data(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler,
                                batch_size=train_batch_size)


    num_train_steps = int(len(train_features) / train_batch_size * num_train_epochs)
    logger = logging.getLogger(__name__)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    # Prepare optimizer and schedule (linear warmup and decay)
    t_total = num_train_steps
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    best_dev_loss = float('inf')
    print('cPosGpt2 training')
    for epoch in trange(int(num_train_epochs), desc="Epoch"):
        avg_loss = 0.
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'labels': batch[1]}

            outputs = model(**inputs)
            loss = outputs[0]
            # loss = model(input_ids, segment_ids, input_mask, masked_ids)
            optimizer.zero_grad()
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            if (step + 1) % 50 == 0:
                print("avg_loss: {}".format(avg_loss / 50))
            # avg_loss = 0.
        # eval on dev after every epoch
        dev_loss = compute_dev_loss(model, dev_dataloader)
        print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print("Saving model. Best dev so far {}".format(best_dev_loss))
            save_model_path = os.path.join(output, 'best_PostagGpt2.pt')
            torch.save(model.state_dict(), save_model_path)

    # augment data using the best model
    
    augment_train_data(model, tokenizer, train_examples,train_label,output,file_name,prefix,max_seq_length,sample_num,temperature,top_k,top_p,repetition_penalty )


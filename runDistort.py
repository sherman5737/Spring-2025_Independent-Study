import numpy as np
import pandas as pd
import torch
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform 
from scipy.stats import pearsonr,spearmanr
import networkx as nx
from io import StringIO
import json

# Import models
import gensim.downloader as api
word2vec_model = api.load("word2vec-google-news-300")

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM
GPTtokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
GPTmodel = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

BERTtokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
BERTmodel = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

T5tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
T5model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

XLNETtokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")
XLNETmodel = AutoModelForCausalLM.from_pretrained("xlnet/xlnet-base-cased")

ALBERTtokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2") 
ALBERTmodel = AutoModelForMaskedLM.from_pretrained("albert/albert-base-v2")

from sentence_transformers import SentenceTransformer
MINImodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
MPNETmodel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
BGEmodel = SentenceTransformer("BAAI/bge-m3")
ROBERTAmodel = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# Load the data
set1 = "Dataset/diverse_safety_adversarial_dialog_350.csv"
set2 = "Dataset/test.jsonl"
set3a = 'Dataset/JBB-Behaviors/data/benign-behaviors.csv'
set3b = 'Dataset/JBB-Behaviors/data/harmful-behaviors.csv'
set4 = 'Dataset/prism-alignment/conversations.jsonl'
set5 = 'Dataset/malicious_instruction.json'
set6 = 'Dataset/wildjailbreak/eval/eval.tsv'
set7 = 'Dataset/GenMO/GenMO_dataset.json'
set8 = 'Dataset/safety-tuned-llamas/data/training/alpaca_small.json'
set9 = 'Dataset/MIC.csv'
set10 = 'Dataset/gest/gest.csv' 

dicesData = pd.read_csv(set1)
jbaData = pd.read_csv(set3a)
jbbData = pd.read_csv(set3b)


with open(set2, 'r', encoding='utf-8') as file:
    json_str = file.read().replace('true', 'True').replace('false', 'False')
json_io = StringIO(json_str)
redditData = pd.read_json(json_io, lines=True)

prismData = pd.read_json(set4, lines=True)

with open(set5, "r", encoding="utf-8") as file:
    SGdata = json.load(file)

wildData = pd.read_csv(set6, sep="\t")

with open(set7, "r", encoding="utf-8") as file:
    genData = json.load(file)

with open(set8, "r", encoding="utf-8") as file:
    safeData = json.load(file)

micData = pd.read_csv(set9)

gestData = pd.read_csv(set10)

# Get unique conversations
unique_value = dicesData.drop_duplicates(subset="item_id")


# Generate embedding of a sentence
#
# Parameters:
# sentence: the sentence to be embedded [string]
# model: the model used to embed the sentence [int] 0 - GPT2, 1 - BERT, 4 - T5, 5 - word2vec, 8 - ALBERT, 9 - XLNET
# Returns:
# sentence_embedding: the embedding of the sentence [numpy array]
def sentence_embed(sentence,model):
    # tokenzie the sentence and put token into model
    if model == 0:
        inputs = GPTtokenizer((sentence), return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = GPTmodel(**inputs, output_hidden_states=True) 

    if model == 1:
        inputs = BERTtokenizer(sentence, return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = BERTmodel(**inputs, output_hidden_states=True)
    
    if model == 4: 
        inputs = T5tokenizer((sentence), return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = T5model.encoder(**inputs, output_hidden_states=True)

    if model == 5:
        words = [word for word in sentence.lower().split() if word in word2vec_model]
        if not words:
            return np.zeros(300)  # Return a zero vector if no words match
        return np.mean([word2vec_model[word] for word in words], axis=0)

    if model == 8: 
        inputs = ALBERTtokenizer((sentence), return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = ALBERTmodel(**inputs, output_hidden_states=True)

    if model == 9: 
        inputs = XLNETtokenizer((sentence), return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = XLNETmodel(**inputs, output_hidden_states=True)
    # get the embeddings 
    hidden_states = outputs.hidden_states  
    
    # get the last layer hidden states
    last_layer_hidden_states = hidden_states[-1] 


    # average the hidden states across all tokens
    sentence_embedding = last_layer_hidden_states.mean(dim=1).squeeze().numpy()  # Averaging across tokens

    return sentence_embedding


# Generate embeddings of whole dataset 
#
# Parameters:
# data: the data to be embedded [list]
# i: the index of the model [int] 0 - GPT2, 1 - BERT, 2 - MPNET, 3 = MINI, 4 = T5, 5 = word2vec, 6 = BGE, 7 = ROBERTA, 8 = ALBERT, 9 = XLNet
#
# Returns:
# embeddings: the embeddings of the data [list]
# # # # # # # # # # # # # # # # # # # # 
def allData(data,i):
    embeddings = []
    for sentence in data:
        sentence = str(sentence)
        if i == 0: #GPT2 Model
            embeddings.append(sentence_embed(sentence,i))
        if i == 1: #BERT Model
            embeddings.append(sentence_embed(sentence,i))
        if i == 2:
            embeddings.append(MPNETmodel.encode(sentence))
        if i == 3:
            embeddings.append(MINImodel.encode(sentence))
        if i == 4: #T5 Model 
            embeddings.append(sentence_embed(sentence,i))
        if i == 5: #word2vec Model
            embeddings.append(sentence_embed(sentence,i))
        if i == 6: 
            embeddings.append(BGEmodel.encode(sentence))
        if i == 7:
            embeddings.append(ROBERTAmodel.encode(sentence))
        if i == 8:  #Albert Model
            embeddings.append(sentence_embed(sentence,i))
        if i == 9:  #XLNet Model
            embeddings.append(sentence_embed(sentence,i))
    return embeddings 


# Calculates the distortion of the two embedding models
#
# Parameters: 
# dataSet: the data to be used [list]
# one: the first model, [int] 0 - GPT2, 1 - BERT, 2 - MPNET, 3 - MINI, 4 - T5, 5 - word2vec, 6 - BGE, 7 - ROBERTA, 8 - ALBERT, 9 - XLNet
# two: the second model [int] 0 - GPT2, 1 - BERT, 2 - MPNET, 3 - MINI, 4 - T5, 5 - word2vec, 6 - BGE, 7 - ROBERTA, 8 - ALBERT, 9 - XLNet
#
# Returns: spearman and p_val 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def diffTwo(dataSet,one,two): 
    embedOne = pd.DataFrame(allData(dataSet,one))
    embedTwo = pd.DataFrame(allData(dataSet,two))
    dist_matrix_One = pdist(embedOne, metric='euclidean')
    dist_matrix_Two = pdist(embedTwo, metric='euclidean')
    spear, pspear = spearmanr(dist_matrix_One, dist_matrix_Two)
    return spear, pspear




#Create whole sentence embeddings datasets 

data1 = redditData['sentences'][0]
data2 = [s.replace("LAMDA:", "").replace("USER:", "") for s in unique_value['context'].dropna().values.tolist()]
data3 = (jbaData['Goal'].values.tolist()) + (jbbData['Goal'].values.tolist()) 
data4 = prismData['opening_prompt']
data5 = [item["query"] for item in SGdata] 
data6 = wildData['adversarial']
data7 = [item["male_story"] for item in genData]  + [item["female_story"] for item in genData]
data8 = [item["instruction"] for item in safeData] 
data9 = micData['Q']
data10 = gestData['sentence']

totalData = [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]
nameData = ['MTEB Dataset','DICE Dataset','JBB Dataset','PRISM Dataset','SG Dataset','WILD Dataset','GEN Dataset', 'SAFE Dataset', 'MIC Dataset', 'GEST Dataset']


# generate the distortion model tables
#
# Parameters:
# batch_num: the dataset to be used [int] 
# 1 - (MTEB, DICES), 2 - (JBB, PRISM), 3 - (SG, WILD), 4 - (GEN, SAFE), 5 - (MIC, GEST)
#
# Returns: NONE
def distortModel(batch_number):
    models = ['GPT2','BERT','MPNET','MINI','T5','word2vec','BGE','ROBERTA','ALBERT','XLNet']
    # 0 = GPT2, 1 = BERT, 2 = MPNET, 3 = MINI, 4 = T5, 5 = word2vec, 6 = BGE, 7 = ROBERTA, 8 = ALBERT, 9 = XLNet
    if batch_number == 1: 
        ran = range(2)
    if batch_number == 2: 
        ran = range(2,4)
    if batch_number == 3:
        ran = range(4,6)
    if batch_number == 4:
        ran = range(6,8)
    if batch_number == 5:
        ran = range(9,10) #exclude dataset MIC
    for td in ran:
        triTable_spear = np.zeros((10, 10))
        triTable_pval = np.zeros((10, 10))
        output_folder = os.path.join(general_folder, nameData[td])
        os.makedirs(output_folder,exist_ok=True)  

        for i in range(10):
            for j in range(i+1,10):
                spear, pspear = diffTwo(totalData[td],i,j)  
                triTable_spear[i,j] = spear 
                triTable_pval[i,j] = pspear
                triTable_spear[j,i] = spear 
                triTable_pval[j,i] = pspear

        # SPEARMAN TABLE 
        fig, ax = plt.subplots(figsize=(6, 4)) 
        ax.set_axis_off()  
        table_spear = plt.table(cellText=np.round(triTable_spear, 3),  
                        loc='center', cellLoc='center',  
                        rowLabels=models, colLabels=models)
        table_spear.auto_set_font_size(False)
        table_spear.set_fontsize(8)
        table_spear.scale(1.2, 1.2)
        plt.title("Spearman Correlation")
        spear_save_path  = os.path.join(output_folder, "spearTable.png")
        plt.savefig(spear_save_path, dpi=300, bbox_inches='tight')  
        plt.close(fig)

        # P_VALUE TABLE
        fig, ax = plt.subplots(figsize=(6, 4))  
        ax.set_axis_off()
        table_pval = ax.table(cellText=np.round(triTable_pval, 3),  
                            loc='center', cellLoc='center',  
                            rowLabels=models, colLabels=models)
        table_pval.auto_set_font_size(False)
        table_pval.set_fontsize(8)
        table_pval.scale(1.2, 1.2)
        plt.title("P-Values")

        pval_save_path = os.path.join(output_folder, "pvalTable.png")
        plt.savefig(pval_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  

general_folder = 'DistortModels'
os.makedirs(general_folder,exist_ok=True)  
nameData = ['MTEB Dataset','DICE Dataset','JBB Dataset','PRISM Dataset','SG Dataset','WILD Dataset','GEN Dataset', 'SAFE Dataset', 'MIC Dataset', 'GEST Dataset']


import argparse

# Main function to run the script, parse arguments and call the run function 
def main():
    parser = argparse.ArgumentParser(description='Generate clusters from datasets')
    parser.add_argument('--batch_size', type=int, required=True, help='Value of batch size, 1 - (MTEB, DICES), 2 - (JBB, PRISM), 3 - (SG, WILD), 4 - (GEN, SAFE), 5 - (MIC, GEST)')
    args = parser.parse_args()
    
    batch_num = args.batch_size
    if batch_num < 0 or batch_num > 5:
        print("Invalid batch size. Please enter a value between 0 and 5.")
        return
    distortModel(batch_num)

if __name__ == "__main__":
    main()


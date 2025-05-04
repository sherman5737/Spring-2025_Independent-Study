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


mtebGraph = ['mtebGPT.gexf','mtebBERT.gexf','mtebMPNET.gexf','mtebMINI.gexf','mtebT5.gexf','mtebword2vec.gexf','mtebBGE.gexf','mtebROBERTA.gexf','mtebALBERT.gexf','mtebXLNet.gexf']
diceGraph = ['diceGPT.gexf','diceBERT.gexf','diceMPNET.gexf','diceMINI.gexf','diceT5.gexf','diceword2vec.gexf','diceBGE.gexf','diceROBERTA.gexf','diceALBERT.gexf','diceXLNet.gexf']
jbbGraph = ['jbbGPT.gexf','jbbBERT.gexf','jbbMPNET.gexf','jbbMINI.gexf','jbbT5.gexf','jbbword2vec.gexf','jbbBGE.gexf','jbbROBERTA.gexf','jbbALBERT.gexf','jbbXLNet.gexf']
prismGraph = ['prismGPT.gexf','prismBERT.gexf','prismMPNET.gexf','prismMINI.gexf','prismT5.gexf','prismword2vec.gexf','prismBGE.gexf','prismROBERTA.gexf','prismALBERT.gexf','prismXLNet.gexf']
sgGraph = ['sgGPT.gexf','sgBERT.gexf','sgMPNET.gexf','sgMINI.gexf','sgT5.gexf','sgword2vec.gexf','sgBGE.gexf','sgROBERTA.gexf','sgALBERT.gexf','sgXLNet.gexf']
wildGraph = ['wildGPT.gexf','wildBERT.gexf','wildMPNET.gexf','wildMINI.gexf','wildT5.gexf','wildword2vec.gexf','wildBGE.gexf','wildROBERTA.gexf','wildALBERT.gexf','wildXLNet.gexf']
genGraph = ['genGPT.gexf','genBERT.gexf','genMPNET.gexf','genMINI.gexf','genT5.gexf','genword2vec.gexf','BGE.gexf','genROBERTA.gexf','genALBERT.gexf','genXLNet.gexf'] 
safeGraph = ['safeGPT.gexf','safeBERT.gexf','safeMPNET.gexf','safeMINI.gexf','safeT5.gexf','safeword2vec.gexf','safeBGE.gexf','safeROBERTA.gexf','safeALBERT.gexf','safeXLNet.gexf']
micGraph = ['micGPT.gexf','micBERT.gexf','micMPNET.gexf','micMINI.gexf','micT5.gexf','micword2vec.gexf','micBGE.gexf','micROBERTA.gexf','micALBERT.gexf','micXLNet.gexf']
gestGraph = ['gestGPT.gexf','gestBERT.gexf','gestMPNET.gexf','gestMINI.gexf','gestT5.gexf','gestword2vec.gexf','gestBGE.gexf','gestROBERTA.gexf','gestALBERT.gexf','gestXLNet.gexf']

totalGraph = [mtebGraph,diceGraph,jbbGraph,prismGraph,sgGraph,wildGraph,genGraph,safeGraph,micGraph,gestGraph]

os.makedirs('GephiGraph', exist_ok=True)

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
        inputs = GPTtokenizer(sentence, return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = GPTmodel(**inputs, output_hidden_states=True) 

    if model == 1:
        inputs = BERTtokenizer(sentence, return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = BERTmodel(**inputs, output_hidden_states=True)
    
    if model == 4: 
        inputs = T5tokenizer(sentence, return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = T5model.encoder(**inputs, output_hidden_states=True)

    if model == 5:
        words = [word for word in sentence.lower().split() if word in word2vec_model]
        if not words:
            return np.zeros(300)  # Return a zero vector if no words match
        return np.mean([word2vec_model[word] for word in words], axis=0)

    if model == 8: 
        inputs = ALBERTtokenizer(sentence,return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = ALBERTmodel(**inputs, output_hidden_states=True)

    if model == 9: 
        inputs = XLNETtokenizer(sentence, return_tensors="pt",truncation=True, max_length=512)
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

# Finds the minimum distance threshold for a fully connected graph using binary search.
#
# Parameters:
# distances: the distances between the embeddings [list]
# embeddings: the embeddings to be used [dataframe]
#
# Returns:
# optimal_threshold: the optimal threshold for the graph [float]

def binary_search_threshold(distances, embeddings):
    # Finds the lowest distance threshold that makes the graph fully connected 
    left, right = 0, len(distances) - 1
    optimal_threshold = distances[-1][2]  # Default to the largest distance if needed

    while left <= right:
        mid = (left + right) // 2
        threshold = distances[mid][2]

        # Create a temporary graph with edges ≤ threshold
        G = nx.Graph()
        for idx in range(len(embeddings)):  
            G.add_node(idx)  # Ensure all nodes are added

        for i, j, dist in distances:
            if dist <= threshold:
                G.add_edge(i, j)

        # Check if the graph is connected
        if nx.is_connected(G):
            optimal_threshold = threshold  
            right = mid - 1
        else:
            left = mid + 1  # Increase the threshold

    return optimal_threshold

# generate a graph using the embeddings
# 
# Parameters:
# embeddings: the embeddings to be used [dataframe]
# ind: the index of the model [int] 0 - GPT2, 1 - BERT, 2 - MPNET , 3 = MINI, 4 = T5, 5 = word2vec, 6 = BGE, 7 = ROBERTA, 8 = ALBERT, 9 = XLNet
# val: the dataset to be used [int] 0 - MTEB, 1 - DICES, 2 - JBB, 3 - PRISM, 4 - SG, 5 - WILD, 6 - GEN, 7 - SAFE, 8 - MIC, 9 - GEST 
#
# Returns:NONE
# gephi graph file saved
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def gephiGenerate(embeddings,ind,val):
    output_folder = 'GephiGraph/' + nameData[val]
    os.makedirs(output_folder, exist_ok=True) 

    pairwise_distances = pdist(embeddings, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)

    distances = sorted([(i, j, distance_matrix[i, j]) for i in range(len(embeddings)) 
                        for j in range(i + 1, len(embeddings))], key=lambda x: x[2])

    optimal_threshold = binary_search_threshold(distances, embeddings)

    G = nx.Graph() 

    for idx, label in enumerate(embeddings.index):
        G.add_node(label)

    for i, j, dist in distances:
            if dist <= optimal_threshold:
                G.add_edge(embeddings.index[i], embeddings.index[j], weight=dist)
    
    print(output_folder)
    nx.draw(G, with_labels=True)

    file_path = os.path.join(output_folder, totalGraph[val][ind])
    nx.write_gexf(G,file_path)
     

# Evaluate the data using different models [GPT2, BERT, MPNET, MINI, T5, word2vec, BGE, ROBERTA, ALBERT, XLNET]
#
# Parameters:
# data: the data to be used [list]
# val: the dataset to be used [int] 0 - MTEB, 1 - DICES, 2 - JBB, 3 - PRISM, 4 - SG, 5 - WILD, 6 - GEN, 7 - SAFE, 8 - MIC, 9 - GEST 
#
# Returns: NONE
# Run completed
# # # # # # # # # # # # # # # # # # # # # # # # # 
def eval(data,val):
    for i in range(0,10):
        embeddings = pd.DataFrame(allData(data,i))
        gephiGenerate(embeddings,i,val)


# Run the dataset through the models and generate word clouds
# 
# Parameters:
# batch_num: the dataset to be used [int] 
# 1 - (MTEB, DICES), 2 - (JBB, PRISM), 3 - (SG, WILD), 4 - (GEN, SAFE), 5 - (MIC, GEST)
#
# Returns:
# Run completed
# # # # # # # # # # # # # # # # # # # # # # # # # # 
def run(batch_num):
    if batch_num == 1: 
        for i in range(2):
            eval(totalData[i],i)
    if batch_num == 2: 
        for i in range(2,4):
            eval(totalData[i],i)
    if batch_num == 3: 
        for i in range(4,6):
            eval(totalData[i],i)
    if batch_num == 4: 
        for i in range(6,8):
            eval(totalData[i],i)
    if batch_num == 5: 
        for i in range(9,10): # excludes mic dataset
            eval(totalData[i],i)

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
    run(batch_num)

if __name__ == "__main__":
    main()




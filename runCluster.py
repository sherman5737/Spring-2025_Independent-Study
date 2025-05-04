import numpy as np
import pandas as pd
import torch
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from io import StringIO
import json
from kneed import KneeLocator

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

DEEPtokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
DEEPmodel = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

from sentence_transformers import SentenceTransformer
MINImodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
MPNETmodel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
BGEmodel = SentenceTransformer("BAAI/bge-m3")
ROBERTAmodel = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

#creating folders where results are to be saved
outmteb1 = "kmean/k-mean_results_mteb/gpt2"
outmteb2 = "kmean/k-mean_results_mteb/bert"
outmteb3 = "kmean/k-mean_results_mteb/mpnet"
outmteb4 = "kmean/k-mean_results_mteb/mini"
outmteb5 = "kmean/k-mean_results_mteb/t5"
outmteb6 = "kmean/k-mean_results_mteb/word2vec"
outmteb7 = "kmean/k-mean_results_mteb/bge"
outmteb8 = "kmean/k-mean_results_mteb/roberta"
outmteb9 = "kmean/k-mean_results_mteb/albert"
outmteb10 = "kmean/k-mean_results_mteb/xlnet"

mtebfolder_kmean = [outmteb1,outmteb2,outmteb3,outmteb4,outmteb5,outmteb6,outmteb7,outmteb8,outmteb9,outmteb10]
for mteb in mtebfolder_kmean:
    os.makedirs(mteb, exist_ok=True)

outdice1 = "kmean/k-mean_results_dice/gpt2"
outdice2 = "kmean/k-mean_results_dice/bert"
outdice3 = "kmean/k-mean_results_dice/mpnet"
outdice4 = "kmean/k-mean_results_dice/mini"
outdice5 = "kmean/k-mean_results_dice/t5"
outdice6 = "kmean/k-mean_results_dice/word2vec"
outdice7 = "kmean/k-mean_results_dice/bge"
outdice8 = "kmean/k-mean_results_dice/roberta"
outdice9 = "kmean/k-mean_results_dice/albert"
outdice10 = "kmean/k-mean_results_dice/xlnet"
dicefolder_kmean = [outdice1,outdice2,outdice3,outdice4,outdice5,outdice6,outdice7,outdice8,outdice9,outdice10]
for dice in dicefolder_kmean:
    os.makedirs(dice, exist_ok=True)

outjbb1 = "kmean/k-mean_results_jbb/gpt2"
outjbb2 = "kmean/k-mean_results_jbb/bert"
outjbb3 = "kmean/k-mean_results_jbb/mpnet"
outjbb4 = "kmean/k-mean_results_jbb/mini"
outjbb5 = "kmean/k-mean_results_jbb/t5"
outjbb6 = "kmean/k-mean_results_jbb/word2vec"
outjbb7 = "kmean/k-mean_results_jbb/bge"
outjbb8 = "kmean/k-mean_results_jbb/roberta"
outjbb9 = "kmean/k-mean_results_jbb/albert"
outjbb10 = "kmean/k-mean_results_jbb/xlnet"
jbbfolder_kmean = [outjbb1,outjbb2,outjbb3,outjbb4,outjbb5,outjbb6,outjbb7,outjbb8,outjbb9,outjbb10]
for jbb in jbbfolder_kmean:
    os.makedirs(jbb, exist_ok=True)

outprism1 = 'kmean/k-means_results_prism/gpt2'
outprism2 = 'kmean/k-means_results_prism/bert'
outprism3 = 'kmean/k-means_results_prism/mpnet'
outprism4 = 'kmean/k-means_results_prism/mini'
outprism5 = 'kmean/k-means_results_prism/t5'
outprism6 = 'kmean/k-means_results_prism/word2vec'
outprism7 = 'kmean/k-means_results_prism/bge'
outprism8 = 'kmean/k-means_results_prism/roberta'
outprism9 = 'kmean/k-means_results_prism/albert'
outprism10 = 'kmean/k-means_results_prism/xlnet'
prismfolder_kmean = [outprism1,outprism2,outprism3,outprism4,outprism5,outprism6,outprism7,outprism8,outprism9,outprism10]
for prism in prismfolder_kmean:
    os.makedirs(prism, exist_ok=True)

outsg1 = 'kmean/k-means_results_sg/gpt2'
outsg2 = 'kmean/k-means_results_sg/bert' 
outsg3 = 'kmean/k-means_results_sg/mpnet'
outsg4 = 'kmean/k-means_results_sg/mini'
outsg5 = 'kmean/k-means_results_sg/t5'
outsg6 = 'kmean/k-means_results_sg/word2vec'
outsg7 = 'kmean/k-means_results_sg/bge'
outsg8 = 'kmean/k-means_results_sg/roberta' 
outsg9 = 'kmean/k-means_results_sg/albert'
outsg10 = 'kmean/k-means_results_sg/xlnet'
sgfolder_kmean = [outsg1,outsg2,outsg3,outsg4,outsg5,outsg6,outsg7,outsg8,outsg9,outsg10]
for sg in sgfolder_kmean: 
    os.makedirs(sg, exist_ok=True)

outwild1 = 'kmean/k-means_results_wild/gpt2'
outwild2 = 'kmean/k-means_results_wild/bert' 
outwild3 = 'kmean/k-means_results_wild/mpnet'
outwild4 = 'kmean/k-means_results_wild/mini'
outwild5 = 'kmean/k-means_results_wild/t5'
outwild6 = 'kmean/k-means_results_wild/word2vec'
outwild7 = 'kmean/k-means_results_wild/bge'
outwild8 = 'kmean/k-means_results_wild/roberta'
outwild9 = 'kmean/k-means_results_wild/albert'
outwild10 = 'kmean/k-means_results_wild/xlnet' 
wildfolder_kmean = [outwild1,outwild2,outwild3,outwild4,outwild5,outwild6,outwild7,outwild8,outwild9,outwild10]
for wild in wildfolder_kmean: 
    os.makedirs(wild, exist_ok=True)

outgen1 = 'kmean/k-means_results_gen/gpt2'
outgen2 = 'kmean/k-means_results_gen/bert'
outgen3 = 'kmean/k-means_results_gen/mpnet'
outgen4 = 'kmean/k-means_results_gen/mini'
outgen5 = 'kmean/k-means_results_gen/t5' 
outgen6 = 'kmean/k-means_results_gen/word2vec'
outgen7 = 'kmean/k-means_results_gen/bge'
outgen8 = 'kmean/k-means_results_gen/roberta' 
outgen9 = 'kmean/k-means_results_gen/albert'
outgen10 = 'kmean/k-means_results_gen/xlnet'
genfolder_kmean = [outgen1,outgen2,outgen3,outgen4,outgen5,outgen6,outgen7,outgen8,outgen9,outgen10]
for gen in genfolder_kmean:
    os.makedirs(gen, exist_ok=True)

outsafe1 = 'kmean/k-means_results_safe/gpt2'
outsafe2 = 'kmean/k-means_results_safe/bert'
outsafe3 = 'kmean/k-means_results_safe/mpnet'
outsafe4 = 'kmean/k-means_results_safe/mini'
outsafe5 = 'kmean/k-means_results_safe/t5'
outsafe6 = 'kmean/k-means_results_safe/word2vec'
outsafe7 = 'kmean/k-means_results_safe/bge' 
outsafe8 = 'kmean/k-means_results_safe/roberta' 
outsafe9 = 'kmean/k-means_results_safe/albert'
outsafe10 = 'kmean/k-means_results_safe/xlnet' 
safefolder_kmean = [outsafe1,outsafe2,outsafe3,outsafe4,outsafe5,outsafe6,outsafe7,outsafe8,outsafe9,outsafe10]
for safe in safefolder_kmean: 
    os.makedirs(safe, exist_ok=True) 

outmic1 = 'kmean/k-means_results_mic/gpt2'
outmic2 = 'kmean/k-means_results_mic/bert'
outmic3 = 'kmean/k-means_results_mic/mpnet' 
outmic4 = 'kmean/k-means_results_mic/mini'
outmic5 = 'kmean/k-means_results_mic/t5' 
outmic6 = 'kmean/k-means_results_mic/word2vec' 
outmic7 = 'kmean/k-means_results_mic/bge'
outmic8 = 'kmean/k-means_results_mic/roberta'
outmic9 = 'kmean/k-means_results_mic/albert'
outmic10 = 'kmean/k-means_results_mic/xlnet'
micfolder_kmean = [outmic1,outmic2,outmic3,outmic4,outmic5,outmic6,outmic7,outmic8,outmic9,outmic10] 
for mic in micfolder_kmean: 
    os.makedirs(mic, exist_ok=True)

outgest1 = 'kmean/k-means_results_gest/gpt2' 
outgest2 = 'kmean/k-means_results_gest/bert'
outgest3 = 'kmean/k-means_results_gest/mpnet' 
outgest4 = 'kmean/k-means_results_gest/mini'
outgest5 = 'kmean/k-means_results_gest/t5'
outgest6 = 'kmean/k-means_results_gest/word2vec' 
outgest7 = 'kmean/k-means_results_gest/bge'
outgest8 = 'kmean/k-means_results_gest/roberta'
outgest9 = 'kmean/k-means_results_gest/albert' 
outgest10 = 'kmean/k-means_results_gest/xlnet'
gestfolder_kmean = [outgest1,outgest2,outgest3,outgest4,outgest5,outgest6,outgest7,outgest8,outgest9,outgest10]
for gest in gestfolder_kmean: 
    os.makedirs(gest, exist_ok=True)

totalKmean = [mtebfolder_kmean,dicefolder_kmean,jbbfolder_kmean,prismfolder_kmean,sgfolder_kmean,wildfolder_kmean,genfolder_kmean,safefolder_kmean,micfolder_kmean,gestfolder_kmean]

inmteb1 = "agglo/agglomeration_results_mteb/gpt2"
inmteb2 = "agglo/agglomeration_results_mteb/bert"
inmteb3 = "agglo/agglomeration_results_mteb/mpnet"
inmteb4 = "agglo/agglomeration_results_mteb/mini"
inmteb5 = "agglo/agglomeration_results_mteb/t5"
inmteb6 = "agglo/agglomeration_results_mteb/word2vec"
inmteb7 = "agglo/agglomeration_results_mteb/bge"
inmteb8 = "agglo/agglomeration_results_mteb/roberta"
inmteb9 = "agglo/agglomeration_results_mteb/albert"
inmteb10 = "agglo/agglomeration_results_mteb/xlnet"
mtebfolder_agglo = [inmteb1,inmteb2,inmteb3,inmteb4,inmteb5,inmteb6,inmteb7,inmteb8,inmteb9,inmteb10]
for mteb in mtebfolder_agglo:
    os.makedirs(mteb, exist_ok=True)

indice1 = "agglo/agglomeration_results_dice/gpt2"
indice2 = "agglo/agglomeration_results_dice/bert"
indice3 = "agglo/agglomeration_results_dice/mpnet"
indice4 = "agglo/agglomeration_results_dice/mini"
indice5 = "agglo/agglomeration_results_dice/t5"
indice6 = "agglo/agglomeration_results_dice/word2vec"
indice7 = "agglo/agglomeration_results_dice/bge"
indice8 = "agglo/agglomeration_results_dice/roberta"
indice9 = "agglo/agglomeration_results_dice/albert"
indice10 = "agglo/agglomeration_results_dice/xlnet"
dicefolder_agglo = [indice1,indice2,indice3,indice4,indice5,indice6,indice7,indice8,indice9,indice10]
for dice in dicefolder_agglo:
    os.makedirs(dice, exist_ok=True)

injbb1 = "agglo/agglomeration_results_jbb/gpt2"
injbb2 = "agglo/agglomeration_results_jbb/bert"
injbb3 = "agglo/agglomeration_results_jbb/mpnet"
injbb4 = "agglo/agglomeration_results_jbb/mini"
injbb5 = "agglo/agglomeration_results_jbb/t5"
injbb6 = "agglo/agglomeration_results_jbb/word2vec"
injbb7 = "agglo/agglomeration_results_jbb/bge"
injbb8 = "agglo/agglomeration_results_jbb/roberta"
injbb9 = "agglo/agglomeration_results_jbb/albert"
injbb10 = "agglo/agglomeration_results_jbb/xlnet"
jbbfolder_agglo = [injbb1,injbb2,injbb3,injbb4,injbb5,injbb6,injbb7,injbb8,injbb9,injbb10]
for jbb in jbbfolder_agglo:
    os.makedirs(jbb, exist_ok=True)

inprism1 = 'agglo/agglomeration_results_prism/gpt2'
inprism2 = 'agglo/agglomeration_results_prism/bert'
inprism3 = 'agglo/agglomeration_results_prism/mpnet'
inprism4 = 'agglo/agglomeration_results_prism/mini'
inprism5 = 'agglo/agglomeration_results_prism/t5'
inprism6 = 'agglo/agglomeration_results_prism/word2vec'
inprism7 = 'agglo/agglomeration_results_prism/bge'
inprism8 = 'agglo/agglomeration_results_prism/roberta'
inprism9 = 'agglo/agglomeration_results_prism/albert'
inprism10 = 'agglo/agglomeration_results_prism/xlnet'
prismfolder_agglo = [inprism1,inprism2,inprism3,inprism4,inprism5,inprism6,inprism7,inprism8,inprism9,inprism10]
for prism in prismfolder_agglo:
    os.makedirs(prism, exist_ok=True)

insg1 = 'agglo/agglomeration_results_sg/gpt2'
insg2 = 'agglo/agglomeration_results_sg/bert' 
insg3 = 'agglo/agglomeration_results_sg/mpnet'
insg4 = 'agglo/agglomeration_results_sg/mini'
insg5 = 'agglo/agglomeration_results_sg/t5'
insg6 = 'agglo/agglomeration_results_sg/word2vec'
insg7 = 'agglo/agglomeration_results_sg/bge'
insg8 = 'agglo/agglomeration_results_sg/roberta' 
insg9 = 'agglo/agglomeration_results_sg/albert'
insg10 = 'agglo/agglomeration_results_sg/xlnet'
sgfolder_agglo = [insg1,insg2,insg3,insg4,insg5,insg6,insg7,insg8,insg9,insg10]
for sg in sgfolder_agglo: 
    os.makedirs(sg, exist_ok=True)

inwild1 = 'agglo/agglomeration_results_wild/gpt2'
inwild2 = 'agglo/agglomeration_results_wild/bert' 
inwild3 = 'agglo/agglomeration_results_wild/mpnet'
inwild4 = 'agglo/agglomeration_results_wild/mini'
inwild5 = 'agglo/agglomeration_results_wild/t5'
inwild6 = 'agglo/agglomeration_results_wild/word2vec'
inwild7 = 'agglo/agglomeration_results_wild/bge'
inwild8 = 'agglo/agglomeration_results_wild/roberta'
inwild9 = 'agglo/agglomeration_results_wild/albert'
inwild10 = 'agglo/agglomeration_results_wild/xlnet' 
wildfolder_agglo = [inwild1,inwild2,inwild3,inwild4,inwild5,inwild6,inwild7,inwild8,inwild9,inwild10]
for wild in wildfolder_agglo: 
    os.makedirs(wild, exist_ok=True)

ingen1 = 'agglo/agglomeration_results_gen/gpt2'
ingen2 = 'agglo/agglomeration_results_gen/bert'
ingen3 = 'agglo/agglomeration_results_gen/mpnet'
ingen4 = 'agglo/agglomeration_results_gen/mini'
ingen5 = 'agglo/agglomeration_results_gen/t5' 
ingen6 = 'agglo/agglomeration_results_gen/word2vec'
ingen7 = 'agglo/agglomeration_results_gen/bge'
ingen8 = 'agglo/agglomeration_results_gen/roberta' 
ingen9 = 'agglo/agglomeration_results_gen/albert'
ingen10 = 'agglo/agglomeration_results_gen/xlnet'
genfolder_agglo = [ingen1,ingen2,ingen3,ingen4,ingen5,ingen6,ingen7,ingen8,ingen9,ingen10]
for gen in genfolder_agglo:
    os.makedirs(gen, exist_ok=True)

insafe1 = 'agglo/agglomeration_results_safe/gpt2'
insafe2 = 'agglo/agglomeration_results_safe/bert'
insafe3 = 'agglo/agglomeration_results_safe/mpnet'
insafe4 = 'agglo/agglomeration_results_safe/mini'
insafe5 = 'agglo/agglomeration_results_safe/t5'
insafe6 = 'agglo/agglomeration_results_safe/word2vec'
insafe7 = 'agglo/agglomeration_results_safe/bge' 
insafe8 = 'agglo/agglomeration_results_safe/roberta' 
insafe9 = 'agglo/agglomeration_results_safe/albert'
insafe10 = 'agglo/agglomeration_results_safe/xlnet' 
safefolder_agglo = [insafe1,insafe2,insafe3,insafe4,insafe5,insafe6,insafe7,insafe8,insafe9,insafe10]
for safe in safefolder_agglo: 
    os.makedirs(safe, exist_ok=True) 

inmic1 = 'agglo/agglomeration_results_mic/gpt2'
inmic2 = 'agglo/agglomeration_results_mic/bert'
inmic3 = 'agglo/agglomeration_results_mic/mpnet' 
inmic4 = 'agglo/agglomeration_results_mic/mini'
inmic5 = 'agglo/agglomeration_results_mic/t5' 
inmic6 = 'agglo/agglomeration_results_mic/word2vec' 
inmic7 = 'agglo/agglomeration_results_mic/bge'
inmic8 = 'agglo/agglomeration_results_mic/roberta'
inmic9 = 'agglo/agglomeration_results_mic/albert'
inmic10 = 'agglo/agglomeration_results_mic/xlnet'
micfolder_agglo = [inmic1,inmic2,inmic3,inmic4,inmic5,inmic6,inmic7,inmic8,inmic9,inmic10] 
for mic in micfolder_agglo: 
    os.makedirs(mic, exist_ok=True)

ingest1 = 'agglo/agglomeration_results_gest/gpt2' 
ingest2 = 'agglo/agglomeration_results_gest/bert'
ingest3 = 'agglo/agglomeration_results_gest/mpnet' 
ingest4 = 'agglo/agglomeration_results_gest/mini'
ingest5 = 'agglo/agglomeration_results_gest/t5'
ingest6 = 'agglo/agglomeration_results_gest/word2vec' 
ingest7 = 'agglo/agglomeration_results_gest/bge'
ingest8 = 'agglo/agglomeration_results_gest/roberta'
ingest9 = 'agglo/agglomeration_results_gest/albert' 
ingest10 = 'agglo/agglomeration_results_gest/xlnet'
gestfolder_agglo = [ingest1,ingest2,ingest3,ingest4,ingest5,ingest6,ingest7,ingest8,ingest9,ingest10]
for gest in gestfolder_agglo: 
    os.makedirs(gest, exist_ok=True)

totalAgglo = [mtebfolder_agglo,dicefolder_agglo,jbbfolder_agglo,prismfolder_agglo,sgfolder_agglo,wildfolder_agglo,genfolder_agglo,safefolder_agglo,micfolder_agglo,gestfolder_agglo]
totalCluster = [totalKmean,totalAgglo]


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
        inputs = ALBERTtokenizer(sentence, return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = ALBERTmodel(**inputs, output_hidden_states=True)

    if model == 9: 
        inputs = XLNETtokenizer(sentence, return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = XLNETmodel(**inputs, output_hidden_states=True)
    if model == 10:
        inputs = DEEPtokenizer(sentence, return_tensors="pt",truncation=True, max_length=512)
        with torch.no_grad():
            outputs = DEEPmodel(**inputs, output_hidden_states=True)
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

# To find the optimal number of clusters
# 
# Parameters:
# embeddings_df: the embeddings to be used [dataframe]
#
# Returns: N Clusters: the optimal number of clusters [int]
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def nCluster(embeddings_df):
    # Determine which range of k to use, 5 - 15
    k_range = range(5, 16)
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings_df)
        inertia.append(kmeans.inertia_) 
        
    kl = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
    best_k = kl.elbow
    return best_k

# Using K-means clustering to cluster the embeddings
#
# Parameters:
# embeddings_df: the embeddings to be used [dataframe]
# ind: the index of the model [int] 0 - GPT2, 1 - BERT, 2 - MPNET, 3 = MINI, 4 = T5, 5 = word2vec, 6 = BGE, 7 = ROBERTA, 8 = ALBERT, 9 = XLNet   
# val: the dataset to be used [int] 0 - MTEB, 1 - DICES, 2 - JBB, 3 - PRISM, 4 - SG, 5 - WILD, 6 - GEN, 7 - SAFE, 8 - MIC, 9 - GEST 
# k_val: the number of clusters [int]
#
# returns:
# cluster_labels: the labels of the clusters 
# k-means plot images saved
def plotKmean(embeddings_df,ind,val,k_val): 
    output_folder = totalKmean[val][ind]
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_df)

    # Perform 10 different k-means clustering 
    for i in range(1,11):
        kmeans = KMeans(n_clusters= k_val)  
        kmeans.fit(embeddings_scaled)
        cluster_labels = kmeans.labels_
        embeddings_df['Cluster'] = cluster_labels

        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c= cluster_labels , cmap='viridis')
        plt.title(f'PCA of KMeans Cluster{i}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        #plt.show()
        file_path = os.path.join(output_folder, f"kmeans_plot_{i}.png")
        plt.savefig(file_path)
        plt.close() 
     
    return cluster_labels

# Cluster the sentences based on their labels into text file
#
# Parameters:
# labels: the labels of the clusters [list]
# sentences: the sentences to be clustered [list]
# ind: the index of the model [int] 0 - GPT2, 1 - BERT, 2 - MPNET, 3 = MINI, 4 = T5, 5 = word2vec, 6 = BGE, 7 = ROBERTA, 8 = ALBERT, 9 = XLNet
# val: the dataset to be used [int] 0 - MTEB, 1 - DICES, 2 - JBB, 3 - PRISM, 4 - SG, 5 - WILD, 6 - GEN, 7 - SAFE, 8 - MIC, 9 - GEST 
# clu: the clustering method used [int] 0 - K-means, 1 - Agglomerative
#
# Returns: NONE
# text files saved in the text folder
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def printLabel(labels,sentences,ind,val,clu):
    output_folder = totalCluster[clu][val][ind]    
    os.makedirs(output_folder, exist_ok=True)

    for label in set(labels):  
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if labels[i] == label]
        
        with open(os.path.join(output_folder, f"cluster_{label}.txt"), "w", encoding="utf-8") as file:
            for sentence in cluster_sentences:
                file.write(str(sentence) + "\n\n") 


# Uses Agglomerative Clustering to cluster the embeddings
# 
# Parameters:
# embeddings: the embeddings to be used [dataframe]
# i: the index of the model [int] 0 - GPT2, 1 - BERT, 2 - MPNET, 3 = MINI, 4 = T5, 5 = word2vec, 6 = BGE, 7 = ROBERTA, 8 = ALBERT, 9 = XLNet
# val: the dataset to be used [int] 0 - MTEB, 1 - DICES, 2 - JBB, 3 - PRISM, 4 - SG, 5 - WILD, 6 - GEN, 7 - SAFE, 8 - MIC, 9 - GEST 
#
# Returns:
# labels: the labels of the clusters
# dendrogram image saved
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def plotAgglomeration(embeddings,i,val):
    # Compute the linkage matrix
    linkage_matrix = linkage(embeddings, method='ward')

    # Calculate distances between merges
    distances = linkage_matrix[:, 2]
    diffs = np.diff(distances)

    # Find the biggest jump in merge distance (elbow)
    max_gap_index = np.argmax(diffs)
    optimal_clusters = len(embeddings) - (max_gap_index + 1)

    # Use the optimal number of clusters
    cluster = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='ward')
    labels = cluster.fit_predict(embeddings)
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Index')
    plt.ylabel('Distance')

    output_folder = totalAgglo[val][i]
    
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, f"dendrogram.png")
    plt.savefig(file_path)
    plt.close()
    return labels

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
    for i in range(10):
        embeddings_df = pd.DataFrame(allData(data,i))
        embeddings = pd.DataFrame(allData(data,i))
        k_val = nCluster(embeddings_df)
        klabel = plotKmean(embeddings_df,i,val,k_val)
        printLabel(klabel,data,i,val,0)
        labels = plotAgglomeration(embeddings,i,val)
        printLabel(labels,data,i,val,1)  


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
        for i in range(9,10):
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
    parser.add_argument('--batch_size', type=int, required=True, help='Value of batch size: 1 - (MTEB, DICES), 2 - (JBB, PRISM), 3 - (SG, WILD), 4 - (GEN, SAFE), 5 - (MIC, GEST)')
    args = parser.parse_args()
    
    batch_num = args.batch_size
    if batch_num < 0 or batch_num > 5:
        print("Invalid batch size. Please enter a value between 0 and 5.")
        return
    run(batch_num) 

if __name__ == "__main__":
    main()


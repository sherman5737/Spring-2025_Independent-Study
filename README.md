# 2025 Independent Study #  

A collection of Python scripts for clustering, word cloud generation, text generation, and graph generation based on text embeddings. 

# Project Structure #
- **runAnalysis.py:** Generate analysis of distortion tables and clusters 
Example: python runAnalysis.py --input [Path to the input text file]

- **runGraph.py :** Generate minimum connected gephi graphs
Example: python runGraph.py  --batch_size [Value of batch size]

- **runDistort.py :** Generate distortion table results
Example: python runDistort.py  --batch_size [Value of batch size]

- **runCluster.py :** Generate clustering results 
Example: python runCluster.py  --batch_size [Value of batch size]


Before running the scripts, make to create a directory named 'Dataset' that have files organized like this:

Dataset 
├── GenMO [1]
│   └── GenMO_dataset.json
├── gest [2]
│   └── gest.csv
├── JBB-Behaviors [3]
│   └── data
│       ├── benign-behaviors.csv 
│       ├── 
│       └── harmful-behaviors.csv 
├── prism-alignment [4]
│   └── conversations.jsonl
├── safety-tuned-llamas [5]
│   └── data
│       └── training
│           └── alpaca_small.json
├── wildjailbreak [6]
│   └── eval
│       └──eval.tsv
├── diverse_safety_adversarial_dialog_350.csv [7]
├── malicious_instruction.json [8]
└── test.jsonl [9]

## Dataset Access

To reproduce our results, you can download the dataset from:

[1] https://github.com/divij30bajaj/GenMO 
[2] https://huggingface.co/datasets/kinit/gest 
[3] https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors 
[4] https://huggingface.co/datasets/HannahRoseKirk/prism-alignment 
[5] https://github.com/vinid/safety-tuned-llamas
[6] https://huggingface.co/datasets/allenai/wildjailbreak
[7] https://github.com/google-research-datasets/dices-dataset/
[8] https://github.com/MurrayTom/SG-Bench 
[9] https://huggingface.co/datasets/mteb/reddit-clustering 

> Note: You may need to request access or agree to a usage license.

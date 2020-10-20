import sys
import json, numpy, os
from gensim.models import Word2Vec
from utils import onto
from utils import BioNLP_Format
from module_predictor import main_predictor
from utils import BioNLP_Format

sys.path.append("..")
from module_train import main_train

sys.setrecursionlimit(100000)

# Automatic load of training data:
mentionsFilePath = "DATA/trainingData/terms_trainObo.json"
attributionsFilePath = "DATA/trainingData/attributions_trainObo.json"
extractedMentionsFile = open(mentionsFilePath , 'r')
dl_trainingTerms = json.load(extractedMentionsFile)
attributionsFile = open(attributionsFilePath, 'r')
attributions = json.load(attributionsFile)

# Load an ontology for your task.
ontobiotiope = onto.loadOnto("DATA/OntoBiotope_BioNLP-ST-2016.obo")

#print(ontobiotiope['OBT:000591'])


with open('../DEMO/DATA/trainingData/terms_train.json') as f:
  onto_terms = json.load(f)

with open('../DEMO/DATA/trainingData/attributions_train.json') as f:
  onto_attributions = json.load(f)

count = 0
for key, value in onto_attributions.items():
        concept_id = onto_attributions[key][0]
        print(onto_terms[key], "-", ontobiotiope[str(concept_id)].name)

#print(onto_attributions['5d4a950'][0][4:])

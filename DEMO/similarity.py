import sys
import json, numpy, os
from gensim.models import Word2Vec
from utils import onto
from utils import BioNLP_Format
from utils import word2term
from module_predictor import main_predictor
from utils import BioNLP_Format
import spacy
from scipy import spatial

sys.path.append("..")
from module_train import main_train

sys.setrecursionlimit(100000)
nlp = spacy.load("en_core_web_sm")

def calculate_similarity(term_1, term_2):
        cosine = 1 - spatial.distance.cosine(term_1, term_2)
        return cosine

# Automatic load of training data:
mentionsFilePath = "DATA/trainingData/terms_trainObo.json"
attributionsFilePath = "DATA/trainingData/attributions_trainObo.json"
extractedMentionsFile = open(mentionsFilePath , 'r')
dl_trainingTerms = json.load(extractedMentionsFile)
attributionsFile = open(attributionsFilePath, 'r')
attributions = json.load(attributionsFile)
ontobiotiope = onto.loadOnto("DATA/OntoBiotope_BioNLP-ST-2016.obo")

def find_head(doc):
  head = None
  if len(doc) > 1:
    for token in doc:
              if token.text == token.head.text:
                head = token.text
  else:
    head = list(doc[0])
  return head

# Load an existing W2V model (Gensim format):
def load_vectors():
          modelPath = "DATA/wordEmbeddings/VST_count0_size100_iter50.model" # the provided models are really small models, just to test execution
          filename, file_extension = os.path.splitext(modelPath)
          print("Loading word embeddings...")
          if file_extension == ".model":
              model = Word2Vec.load(modelPath)
              word_vectors = dict((k, list(numpy.float_(npf32) for npf32 in model.wv[k])) for k in model.wv.vocab.keys()) # To improve, to take directly a binary model from Gensim.
              #del model
          elif file_extension == ".json":
              VSTjsonFile = open(modelPath, 'r')
              word_vectors = json.load(VSTjsonFile)

          return model, word_vectors

model, word_vectors = load_vectors()
print("Word embeddings loaded.\n")

vstTerm, unknownTokens = word2term.wordVST2TermVST(word_vectors, dl_trainingTerms)
#print("Unknown tokens (possibly tokens from labels of the ontology): "+str(unknownTokens))

with open('../DEMO/DATA/trainingData/terms_train.json') as f:
  onto_terms = json.load(f)

with open('../DEMO/DATA/trainingData/attributions_train.json') as f:
  onto_attributions = json.load(f)

count = 0
for key, value in onto_attributions.items():
        concept_id = onto_attributions[key][0]
        doc = nlp(ontobiotiope[str(concept_id)].name)
        if((find_head(doc) in model.wv.vocab) and (onto_terms[key][0] in model.wv.vocab)):
          if(isinstance(onto_terms[key], list) and len(onto_terms[key]) > 1):
            #cosine similarity
            cosine = calculate_similarity(model[find_head(doc)], model[onto_terms[key][0]])
            #cosine = 1 - spatial.distance.cosine(model[find_head(doc)], model[onto_terms[key][0]])
            print(cosine)

            #print("Heads: ", find_head(doc), onto_terms[key][0])
            #print(onto_terms[key], "-", ontobiotiope[str(concept_id)].name)

# -*- coding: utf-8 -*-
"""
@author: payam.bagheri

The idea here is to take the qual responses and try to connect respondents with
 each other using network analysis based on the words that they use in their 
 responses.
"""

# =============================================================================
# Libraries
# =============================================================================
import networkx as nx
import pandas as pd
import numpy as np
from os import path
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# Inputs
# =============================================================================
dir_path = path.dirname(path.dirname(path.abspath(__file__)))
print(dir_path)

data = pd.read_excel(dir_path + '/0_input_data/2090 - XP Manifesto - Content Analysis v02.xlsx')
data.columns

# =============================================================================
# Functions
# =============================================================================
def text_prep(mess):
    tokens = word_tokenize(str(mess))
    words = [word.lower() for word in tokens if word.isalnum()]
    words = [w for w in words if not w in stop_words]
    return words

# =============================================================================
# Data Exploration and preparation
# =============================================================================
data.Author.unique()
len(data.Author.unique())
# Matt and Simon are moderators, so we'll leave them out of the analysis

resps = list(data.Author.unique())[2:]
len(resps)

resp_stenteces = pd.DataFrame(columns = ['sentences', 'words', 'num_words', 'num_imp_words' ,'tfidf'], index = resps)

for r in resps:
    mod_sents = data.Message[data.Author == r][1:-1].apply(str)
    sents = mod_sents.str.cat(sep=' ') #[1:-1] is to remove the enter and leave chat statements
    clean_sents = text_prep(sents)
    resp_stenteces['sentences'].loc[r] = clean_sents


'''
corpus = [
    'This is the first document.',
    'This document is the second document. payam',
    'And this is the third one.',
    'Is this the first document?']
'''

# I need to add misspelling correction

corpus = list(resp_stenteces['sentences'].apply(lambda x: ' '.join(x)))

tf_idf_vect = TfidfVectorizer()
final_tf_idf = tf_idf_vect.fit_transform(corpus)
tfidf_feat = tf_idf_vect.get_feature_names()

len(tfidf_feat)


word_tfidf = pd.DataFrame(final_tf_idf.toarray())
word_tfidf.columns = tfidf_feat 

for i in word_tfidf.index: 
    temp = pd.DataFrame(word_tfidf.iloc[i])
    temp = temp.sort_values(by=[i],ascending = False)
    temp = temp[temp[i] > 0.1]
    temp['words'] = temp.index
    temp['tups'] = list(zip(temp['words'], temp[i]))
    resp_stenteces['words'].iloc[i] = list(temp['words'])
    resp_stenteces['num_words'].iloc[i] = len(resp_stenteces['sentences'].iloc[i])
    resp_stenteces['num_imp_words'].iloc[i] = len(list(temp['words']))
    resp_stenteces['tfidf'].iloc[i] = list(temp['tups'])




'''
#Applying TF-IDF scores to the model vectors
tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
errors=0
for sent in tqdm(tokens): # for each review/sentence
    sent_vec = np.zeros(100) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf [row, tfidf_feat.index(word)]
            sent_vec += (vec * tfidf)
            weight_sum += tfidf
        except:
            errors =+1
            pass
    sent_vec /= weight_sum
    #print(np.isnan(np.sum(sent_vec)))
    tfidf_sent_vectors.append(sent_vec)
    row += 1
    print('errors noted: '+str(errors))
'''

# =============================================================================
# Network Analysis
# =============================================================================
# ideas: connect people who use words with related meanings (using word2vec, or a more suitable embedded vector set)?






# =============================================================================
# Outputs
# =============================================================================
resp_stenteces.to_csv(dir_path + '/0_output/2090_qual_data.csv')
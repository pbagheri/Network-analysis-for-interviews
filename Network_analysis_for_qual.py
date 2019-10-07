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
guide = pd.read_excel(dir_path + '/0_input_data/2090 - XP Gamer - Qual Guide v01.xlsx', header=None)
guide.columns

# =============================================================================
# General Functions
# =============================================================================
def text_prep(mess):
    tokens = word_tokenize(str(mess))
    words = [word.lower() for word in tokens if word.isalnum()]
    words = [w for w in words if not w in stop_words]
    return words

# determines whetehr a string contains a character 
def is_in(ch,st):
   return not(ch not in st) 

# =============================================================================
# Data Exploration and preparation
# =============================================================================
data.Author.unique()
len(data.Author.unique())
# Matt and Simon are moderators, so we'll leave them out of the analysis

moderator_mess = data['Message'][data['Author Type'] == 'Moderator']
moderator_mess.index

respondent_mess = data['Message'][data['Author Type'] == 'Respondent']
respondent_mess.index

moderator_questions = moderator_mess[moderator_mess.apply(lambda x: is_in('?',x))]
moderator_questions.index
moderator_questions.shape





for i in range(len(moderator_questions.index)):
    if i != len(moderator_questions.index) - 1:
        ans_range_inds = list(range(moderator_questions.index[i]+1,moderator_questions.index[i+1]))
        print(ans_range_inds)
    else:
        ans_range_inds = list(range(moderator_questions.index[i]+1,10000))
        print(ans_range_inds)
        
        
moderator_questions.unique().shape
moderator_questions.unique
unique_questions_indeces = []
for i in moderator_questions.index:
    inds = []

resps = list(data.Author.unique())[2:]
len(resps)

resp_sentences = pd.DataFrame(columns = ['raw_sentences','clean_sentences', 'words', 'num_words', 'num_sig_words' ,'tfidf'], index = resps)

for r in resps:
    mod_sents = data.Message[data.Author == r][1:-1].apply(str)
    sents = mod_sents.str.cat(sep=' ') #[1:-1] is to remove the enter and leave chat statements
    resp_sentences['raw_sentences'].loc[r] = sents
    clean_sents = text_prep(sents)
    resp_sentences['clean_sentences'].loc[r] = clean_sents


'''
corpus = [
    'This is the first document.',
    'This document is the second document. payam',
    'And this is the third one.',
    'Is this the first document?']
'''

# I need to add misspelling correction

corpus = list(resp_sentences['clean_sentences'].apply(lambda x: ' '.join(x)))

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
    resp_sentences['words'].iloc[i] = list(temp['words'])
    resp_sentences['num_words'].iloc[i] = len(resp_sentences['clean_sentences'].iloc[i])
    resp_sentences['num_sig_words'].iloc[i] = len(list(temp['words']))
    resp_sentences['tfidf'].iloc[i] = list(temp['tups'])

resp_sentences['sig_word_ratio'] = resp_sentences['num_sig_words']/resp_sentences['num_words']



# =============================================================================
# Network Analysis
# =============================================================================
# ideas: connect people who use words with related meanings (using word2vec, or a more suitable embedded vector set)?






# =============================================================================
# Outputs
# =============================================================================
resp_sentences.to_csv(dir_path + '/0_output/2090_qual_data.csv')

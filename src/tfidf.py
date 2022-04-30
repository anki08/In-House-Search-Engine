# Imports

import pandas as pd
import numpy as np
import sys
import os 
import re
import operator
import nltk 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

# Parse text data

def parseJSON(file):
    news = pd.read_json(file)
    
    for i, txt in enumerate(news['content']):
        subject = re.findall('Subject:(.*\n)',txt)
        if (len(subject) !=0):
            news.loc[i,'Subject'] =str(i)+' '+subject[0]
        else:
            news.loc[i,'Subject'] ='NA'
            
    news = news[['Subject', 'content']]
    
    return news


# Data cleaning

def cleanData(data):
    data.Subject = data.Subject.replace(to_replace='\n', value='', regex=True)
    data.content = data.content.replace(to_replace='From:(.*\n)',value='',regex=True)
    data.content = data.content.replace(to_replace='Lines:(.*\n)',value='',regex=True)
    data.content = data.content.replace(to_replace='[!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]',value=' ',regex=True)
    data.content = data.content.replace(to_replace='-',value=' ',regex=True)
    data.content = data.content.replace(to_replace='\s+',value=' ',regex=True)
    data.content = data.content.replace(to_replace='  ',value='',regex=True)  
    data.content = data.content.apply(lambda x:x.strip())
    data.loc[:, 'content'] = data['content'].str.lower()
    
    return data


# Word lemmatize
    
def lemmatizeWords(data):
    d = defaultdict(lambda: wn.NOUN)
    d['J'] = wn.ADJ
    d['V'] = wn.VERB
    d['R'] = wn.ADV
    stopwords_set = set(stopwords.words('english'))
    clean_data = pd.DataFrame()

    for i, sentence in enumerate(data):
        words = []
        lemmatizer = WordNetLemmatizer()
        # Use NLTK to check if the word is an adjective, noun or verb
        for word, tag in pos_tag(sentence.split()):
            if len(word) > 1 and word not in stopwords_set and word.isalpha():
                lemmatized_word = lemmatizer.lemmatize(word, d[tag[0]])
                words.append(lemmatized_word)
        
        clean_data.loc[i, 'lemmatized'] = str(words)
        
    clean_data = clean_data.replace(to_replace ="\[.", value = '', regex = True)
    clean_data = clean_data.replace(to_replace ="'", value = '', regex = True)
    clean_data = clean_data.replace(to_replace =" ", value = '', regex = True)
    clean_data = clean_data.replace(to_replace ="\]", value = '', regex = True)
        
    return clean_data


# Get TF-IDF weights of the whole vocabulary

def createTFIDF(wordsList):
    vocabulary = set()

    for words in wordsList:
        vocabulary.update(words.split(','))

    for i in range(50):                     # Taking first 50
        vocabulary.update([i].split(','))

    vocabulary = list(vocabulary)
    tfidf = TfidfVectorizer(vocabulary=vocabulary)
    tfidf.fit(wordsList)
    tfidf_normalized = tfidf.transform(wordsList)
    
    return vocabulary, tfidf, tfidf_normalized


# Vector for search keywords

def vector_keywords(tokens, vocabulary, tfidf):
    queries = np.zeros((len(vocabulary)))
    x = tfidf.transform(tokens)
    for token in tokens[0].split(','):
        try:
            i = vocabulary.index(token)
            queries[i] = x[0, tfidf.vocabulary_[token]]
        except:
            pass
        
    return queries


# Cosine similarity between documents

def cosine_similarity(keyword, vocabulary, tfidf, tfidf_normalized, news, k=5):
    result = pd.DataFrame()
    cosines = []
    df = pd.DataFrame(columns=['queries'])
    preprocessed_query = re.sub("\W+", " ", keyword).strip()
    tokens = word_tokenize(str(preprocessed_query))
    tokens = ' '.join(tokens)
    
    df.loc[0, 'queries'] = tokens
    df['queries'] = lemmatizeWords(df['queries'])
    
    query_vector = vector_keywords(df['queries'], vocabulary, tfidf)

    for d in tfidf_normalized.A:
        cos_sim = np.dot(query_vector, d) / (np.linalg.norm(query_vector)*np.linalg.norm(d))
        cosines.append(cos_sim)
    
    out = np.array(cosines).argsort()[-k:][::-1]
    cosines.sort()
    
    for i, index in enumerate(out):
        result.loc[i, 'index'] = str(index)
        res = news['Subject'][index].split()
        res = ' '.join(res[1:])
        result.loc[i, 'Subject'] = res
        result.loc[i, 'Content'] = news['content'][index]
        result.Content = result.Content.replace(to_replace ="(subject re |subject )", value = '', regex = True)
    for j, score in enumerate(cosines[-k:][::-1]):
        result.loc[j, 'Score'] = score
        
    return result


# file = 'https://raw.githubusercontent.com/zayedrais/DocumentSearchEngine/master/data/newsgroups.json'
# news = parseJSON(file)
# news = cleanData(news)
# news_lemmatized = lemmatizeWords(news['content'])
# news = pd.concat([news, news_lemmatized], axis=1)
# news.to_json(r'preprocessedData.json')

news = pd.read_json('preprocessedData.json')
vocabulary, tfidf, tfidf_normalized = createTFIDF(news['lemmatized'])

print(cosine_similarity('computer science', vocabulary, tfidf, tfidf_normalized, news, 10))
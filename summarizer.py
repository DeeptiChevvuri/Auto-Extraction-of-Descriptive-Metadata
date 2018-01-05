#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:15:47 2017

@author: deeptichevvuri
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim
import re
import os
from summarizehelper import FrequencySummarizer


tokenizer = RegexpTokenizer(r'\w+')

# importing  stop words list
with open('/Users/deeptichevvuri/Documents/CC/data/stop words.txt','r') as input_buffer:
    en_stop=[]
    for line in input_buffer:
        en_stop.append(line.strip())

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# loop through document list
#for i in doc_set:
a_dir='/Users/deeptichevvuri/Documents/CC/data/Download3'
categoryTopics={}
groupTopics=[]
for name in os.listdir(a_dir):
    if os.path.isdir(os.path.join(a_dir, name)):
        groupTopics.append(name)
        i=""
        for file in os.listdir(os.path.join(a_dir, name)): 
            org_file = os.path.join(a_dir, name)+'/'+file
            if file=='.DS_Store':
                continue
            with open(org_file, 'r') as myFile:
                for line in myFile:
                    i=i+line.strip()+' ' 
        texts = []
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stemmed_tokensfinal=[]
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        for word in stemmed_tokens:
            if len(word)>3:
                stemmed_tokensfinal.append(word)
                # add tokens to list
        texts.append(stemmed_tokensfinal)
        # turn our tokenized documents into a id <-> term dictionarysssss
        dictionary = corpora.Dictionary(texts)
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word=dictionary, passes=20)
        myList=ldamodel.print_topics(num_topics=1, num_words=20)
        myList2=myList[0]
        categoryTopics[name]=re.findall(r'"([^"]*)"', myList2[1])
        print("Category- "+name+":")
        print(categoryTopics[name])

#test data
#for i in doc_set:
a_dir='/Users/deeptichevvuri/Documents/CC/data/testdata'
for foldername in os.listdir(a_dir):
   if os.path.isdir(os.path.join(a_dir, foldername)):
       i=""
       with open('/Users/deeptichevvuri/Documents/CC/data/output.txt','a') as input_buffer:
           input_buffer.write("Original Group\t"+"Classified Group\t"+" Similarity Score\t "+"Correct Classification\n")   
       for file in os.listdir(os.path.join(a_dir, foldername)): 
           documentTopics={}
           org_file = os.path.join(a_dir, foldername)+'/'+file
           if file=='.DS_Store':
               continue
           with open(org_file, 'r') as myFile:
               for line in myFile:
                   i=i+line.strip()+' ' 
           texts = []
           # clean and tokenize document string
           raw = i.lower()
           tokens = tokenizer.tokenize(raw)
           stemmed_tokensfinal=[]
           # remove stop words from tokens
           stopped_tokens = [i for i in tokens if not i in en_stop]
           # stem tokens
           stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
           for word in stemmed_tokens:
               if len(word)>3:
                   stemmed_tokensfinal.append(word)
           # add tokens to list
           texts.append(stemmed_tokensfinal)
           # turn our tokenized documents into a id <-> term dictionarysssss
           dictionary = corpora.Dictionary(texts)
           # convert tokenized documents into a document-term matrix
           corpus = [dictionary.doc2bow(text) for text in texts]
           # generate LDA model
           ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word=dictionary, passes=20)
           myList=ldamodel.print_topics(num_topics=1, num_words=20)
           myList2=myList[0]
           documentTopics[foldername+'/'+file]=re.findall(r'"([^"]*)"', myList2[1])
           similarityScore=0.0
           finalGroup=''
           classification='True'
           for groupType in groupTopics:
               similarityIndex=0
               for docTop in documentTopics[foldername+'/'+file]:
                   for catTop in categoryTopics[groupType]:
                       if docTop==catTop:
                           similarityIndex=similarityIndex+documentTopics[foldername+'/'+file].index(docTop)*categoryTopics[groupType].index(catTop)
               #print(similarityIndex/28.7)
               currentSimmilarityScore=similarityIndex/28.7
               if currentSimmilarityScore>similarityScore:
                   similarityScore=currentSimmilarityScore
                   finalGroup=groupType
           if finalGroup!=foldername:
               #print('wrong classification')
               classification='False'
           with open('/Users/deeptichevvuri/Documents/CC/data/output.txt','a') as input_buffer:
               input_buffer.write(foldername+"\t"+finalGroup+"\t"+str(similarityScore)+"\t")   
               input_buffer.write(classification+"\n")
           if finalGroup==foldername:
               summayTopics=categoryTopics[finalGroup]
               fs = FrequencySummarizer()
               print("summary")
               print('*************************************************************************************')
               for s in fs.summarize(raw, 2,summayTopics):
                   print('*',s)

        


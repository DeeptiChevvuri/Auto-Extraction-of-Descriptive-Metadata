from nltk.tokenize import sent_tokenize,word_tokenize
from collections import defaultdict
from heapq import nlargest

class FrequencySummarizer:
  def __init__(self, mincut=0.1, maxcut=0.9):
   
    with open('/Users/deeptichevvuri/Documents/CC/data/stop words.txt','r') as input_buffer:
        en_stop=[]
        for line in input_buffer:
            en_stop.append(line.strip())
    self._stopwords = en_stop

  def summaryTopics(self, word_sent, words):
  
    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in self._stopwords:
          freq[word] += 1
    # filtering topic words only
    m = float(max(freq.values()))
    for w in list(freq):
        if w in words:  
            freq[w] = freq[w]/m
        else:
            del freq[w]
    return freq
  def summarize(self, text, n, words):
    """
     creates the summary using the topic words
    """
    sents = sent_tokenize(text)
    assert n <= len(sents)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    #print(word_sent)
    self._freq = self.summaryTopics(word_sent, words)
    print(self._freq)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in self._freq:
          ranking[i] += self._freq[w]
    sents_idx = self.sentenceRanking(ranking, n)    
    return [sents[j] for j in sents_idx]

  def sentenceRanking(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)


import numpy as np
import re
import pandas as pd

class custom_vsm:
  def __init__(self,corpus, stopwords = None, use_idf = False, retrieve_count = None):
    if stopwords is None:
      self.stopwords = {'a', 'an' ,'the', 'of', 'in' , 'on', 'is', 'and', 'from',
      'but','to','into','below','above','under','over','between','among'}
    else:
      self.stopwords = stopwords

    if retrieve_count is None:
      self.retrieve_count = 3
    else:
      self.retrieve_count = retrieve_count

    self.use_idf = use_idf
    #print ("Stopwords:")
    #print (self.stopwords)
    #Corpus as a list of documents
    self.corpus = corpus

    #print ("Initial Corpus:")
    #print (self.corpus)
    #Preprocessing the corpus by removing punctuations and case folding by converting to lower case
    self.preprocess_corpus()
    #print ("Preprocessed corpus:")
    #print (self.corpus)

    self.vocab = self.get_vocab()        #set of unique words in corpus ignoring stopword
    self.N = len(self.vocab)     #vocabulary length

    #print ("Vocab:")
    #print (self.vocab)

    self.term_idfs = {}  # storing the idf score of each term with respect to documents
    # Term document matrix
    self.arr = self.get_term_document_matrix() #d*V
    #print ("Term document Matrix:")
    #print (self.arr)

  def preprocess_corpus(self):
    for i,document in enumerate(self.corpus):
      self.corpus[i] = self.preprocess_text(document)

  def preprocess_text(self,text):

    punctuations = re.compile(r'[,:\'\"\?\!.\n()]')
    preprocessed_text = re.sub(punctuations,'', text)
    #preprocessed_text = re.sub(r"\'s",'', preprocessed_text)
    preprocessed_text = re.sub(r'[\-;]',' ', preprocessed_text)


    return (preprocessed_text.lower())

  def get_vocab(self):
    vocab = set()
    for document in self.corpus:
      words = document.split()
      for word in words:
        if word not in self.stopwords and word!=' ':
            vocab.add(word)




    return list(vocab)


  def get_term_document_matrix(self):
    idf_scale = self.use_idf
    arr = []
    for i,document in enumerate(self.corpus):
      #document = self.preprocess(document)
      words = document.split()

      term_counts = {}
      for word in words:
        term_counts[word] = term_counts.get(word,0) + 1

      doc_vector = [0] * self.N
      for j,term in enumerate(self.vocab):
        doc_vector[j] = term_counts.get(term,0)

      arr.append(doc_vector)
      for term in term_counts.keys():
        self.term_idfs[term] = self.term_idfs.get(term,0) + 1

    d = len(self.corpus)
    for term in self.term_idfs.keys():
      self.term_idfs[term] = np.log((d/self.term_idfs[term]))

    if idf_scale==True:
      for i in range(d):
        for j in range(self.N):
          term = self.vocab[j]
          idf_score = self.term_idfs[term]

          arr[i][j] = arr[i][j] * idf_score

    arr = np.array(arr)
    return arr

  def cosine_score(self,q, d):
    n_q = np.linalg.norm(q,2)
    n_d = np.linalg.norm(d,2)

    score = np.dot(q,d)/(n_q*n_d)
    return score

  def query2vec(self, query):
    q = [0] * self.N

    query = self.preprocess_text(query).split()
    term_count = {}
    for word in query:
      term_count[word] = term_count.get(word,0) + 1

    for i in range(self.N):
      term = self.vocab[i]
      q[i] = term_count.get(term,0)

    return q


  def search(self, query):

    q = self.query2vec(query)

    doc_scores = []
    num_docs = len(self.corpus)
    for i in range(num_docs):
      d = self.arr[i]
      score = self.cosine_score(q,d)
      doc_scores.append((i,score))


    doc_scores.sort(key = lambda x:x[1], reverse = True)


    return (doc_scores[:self.retrieve_count])

import numpy as np
import re
import pandas as pd

class context_vsm:
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
    self.D = self.arr.shape[0]
    #print ("Term document Matrix:")
    #print (self.arr)
    self.contexts = self.get_context()

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
    term_count = {}
    for word in query:
      term_count[word] = term_count.get(word,0) + 1

    for i in range(self.N):
      term = self.vocab[i]
      q[i] = term_count.get(term,0)

    return q

  def search(self, query):

    query = self.preprocess_text(query).split()
    q = self.query2vec(query)
    query_context = self.get_query_context(query)

    if query_context!=-1:   #context of query found
        c = self.possible_contexts[query_context]
        #print ("Context for query \"%s\" is %s" %(' '.join(query), c))
        print ("Context found for query is: {}\n".format(c))
        relevant_docs = self.contexts[c]
        non_rel_docs = list(set(range(self.D)) - set(relevant_docs))
        q = self.rochio_feedback(q, relevant_docs, non_rel_docs)
    else:
        print ("Context of query not found..Proceeding without Rocchio feedback:")
        pass  #context of query not found




    doc_scores = []
    #num_docs = len(self.corpus)
    #num_docs = self.D
    for i in range(self.D):
      d = self.arr[i]
      score = self.cosine_score(q,d)
      doc_scores.append((i,score))


    doc_scores.sort(key = lambda x:x[1], reverse = True)


    return (doc_scores[:self.retrieve_count])


  def get_context(self):

      self.possible_contexts = ['Rank', 'Nature','Broadcast','Flight']
      num_docs = self.arr.shape[0]
      term_contexts = pd.read_csv('prob_term_context.csv')
      self.context_words = set(term_contexts['Words'].values)
      self.doc_contexts = np.zeros((num_docs, len(self.possible_contexts)))
      #print (len(self.context_words))
      for i in range(num_docs):
          prob_vector = np.zeros(len(self.possible_contexts))
          c = 0
          for j in range(len(self.vocab)):
              if self.arr[i,j]>0:
                  if self.vocab[j] in self.context_words:
                      #print (self.vocab[j])
                      t = term_contexts.loc[term_contexts['Words']==self.vocab[j]]
                      t = np.array(t[self.possible_contexts].values)

                      #print (t)
                      #context = np.array(t[1:])
                      #print (context)
                      prob_vector = prob_vector + t
                      c+=1

          if c!=0:
              prob_vector = prob_vector/c
          else:
              print ("Context not found for document %d" %i)

          #print (prob_vector.shape)
          self.doc_contexts[i] = prob_vector

      #print (np.argmax(self.doc_contexts, axis=1))
      #self.contexts.append(np.argmax(prob_vector, axis = 1))
      t = np.argmax(self.doc_contexts, axis = 1)
      contexts = [self.possible_contexts[i] for i in t]
      doc_contexts = {}
      for context in self.possible_contexts:
          doc_contexts[context] = []

      for i, context in enumerate(contexts):
          doc_contexts[context].append(i)

      return doc_contexts

  def get_query_context(self,query):
      context = -1
      term_contexts = pd.read_csv('prob_term_context.csv')
      prob_vector = np.zeros(len(self.possible_contexts))
      c = 0
      for word in query:
          #print (self.vocab[j])
          if word in self.context_words:
              t = term_contexts.loc[term_contexts['Words']==word]
              t = np.array(t[self.possible_contexts].values)

          #print (t)
          #context = np.array(t[1:])
          #print (context)
              prob_vector = prob_vector + t
              c+=1

      if c!=0:
          #k = 0
          #for
          prob_vector = prob_vector/c
          context = np.argmax(prob_vector)
      else:
          pass
      return context

  def rochio_feedback(self, q, relevant_docs, non_rel_docs,alpha = 1, beta = 0.8, gamma = 0.2):
      rel_d = np.zeros(self.N)
      for doc in relevant_docs:
          d = self.arr[doc]
          rel_d += d

      rel_d = rel_d/len(relevant_docs)

      non_rel_d = np.zeros(self.N)
      for doc in non_rel_docs:
          d = self.arr[doc]
          non_rel_d += d

      non_rel_d = non_rel_d/len(non_rel_d)

      q = (alpha*q) + (beta*rel_d) - (gamma*non_rel_d)
      return q

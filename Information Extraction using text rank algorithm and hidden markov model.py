#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')


# In[2]:


import re

import numpy as np
from nltk import sent_tokenize, word_tokenize

from nltk.cluster.util import cosine_distance

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


# In[3]:


def normalize_whitespace(text):
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)
def _replace_whitespace(match):
    text = match.group()

    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "
def is_blank(string):
    return not string or string.isspace()


# In[4]:


def get_symmetric_matrix(matrix):
    
    return matrix + matrix.T - np.diag(matrix.diagonal())


# In[5]:


def core_cosine_similarity(vector1, vector2):
   
    return 1 - cosine_distance(vector1, vector2)


# In[6]:


class TextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps
        self.text_str = None
        self.sentences = None
        self.pr_vector = None

    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return core_cosine_similarity(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)

        # Get Symmeric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number=5):

        top_sentences = []

        if self.pr_vector is not None:

            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for epoch in range(number):
                sent = self.sentences[sorted_pr[index]]
                sent = normalize_whitespace(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)

        self.pr_vector = self._run_page_rank(similarity_matrix)


# In[28]:


text_str = '''
   Computer Science Computer science is one of the fastest growing career fields in modern history. Dating back only a few decades to the late 1950's and early 1960's, it has become on of the leading industries in the world today. Developed through the technological architecture of electrical engineering and the computational language of mathematics, the science of computer technology has provided considerable recognition and financial gain for many of its well deserving pioneers. Originally conceived as an organizational solution to the massive amounts of information kept on nothing more than paper, computers have evolved and advanced to become a common part of modern day life. In the early days of the computer age, the newest and most…show more content…
It is essentially the brain of the computer and though it is the main determining factor in the processing power of the computer as a whole, many other parts of the machine are just as important in overall performance. Many people don't know this and that is how computer corporations have cheated people out of their money for so many years by selling them cheep systems with high megahertz numbers for the processors in them. This is one reason for the success of the computer industry. When people find out that they have been cheated, they will try to learn more about the product and probably end up spending more money next time. Either way the computer companies always win. A career in the field of computer science has been proven to be a worthwhile direction for any young enthusiast and this tren is looking just as bright in the new millenium. Computer science and technology has much to offer in anyone of its many career paths. Whether working with a large multinational corporation or a smaller private company on computer hardware or software in engineering or programming, the possibilities and opportunities are endless and are increasing everyday.

    '''


# In[29]:


pip install spacy


# In[30]:


import spacy


# In[31]:


nlp = spacy.load('en_core_web_sm')
doc = nlp(text_str)


# In[32]:


for token in doc:
    print(token.text,'->',token.pos_)


# In[33]:


from spacy import displacy 
displacy.render(doc, style='dep',jupyter=True)


# In[34]:


##   HMMM IMPLEMENTATION 


# In[35]:


import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time
nltk.download('treebank')
nltk.download('universal_tagset')
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
print(nltk_data[:5])


# In[36]:


for sent in nltk_data[:5]:
  for tuple in sent:
    print(tuple)


# In[37]:


train_set,test_set =train_test_split(nltk_data,train_size=0.80,test_size=0.20,random_state = 101)


# In[38]:


train_tagged_words = [ tup for sent in train_set for tup in sent ]
test_tagged_words = [ tup for sent in test_set for tup in sent ]
print(len(train_tagged_words))
print(len(test_tagged_words))


# In[39]:


train_tagged_words[:15]


# In[40]:


tags = {tag for word,tag in train_tagged_words}
print(len(tags))
print(tags)
vocab = {word for word,tag in train_tagged_words}


# In[41]:


def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)


# In[42]:


def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# In[43]:


tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
 
print(tags_matrix)


# In[44]:


tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
display(tags_df)


# In[ ]:


## Viterbi Algorithm for optimization


# In[45]:


def Viterbi(words, train_bag = train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    for key, word in enumerate(words):
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
        pmax = max(p)
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))


# In[46]:


random.seed(1234) 
rndom = [random.randint(1,len(test_set)) for x in range(10)]
test_run = [test_set[i] for i in rndom]
test_run_base = [tup for sent in test_run for tup in sent]
test_tagged_words = [tup[0] for sent in test_run for tup in sent]


# In[47]:


start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end-start 
print("Time taken in seconds: ", difference)
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)


# In[48]:


patterns = [
    (r'.*ing$', 'VERB'),              # gerund
    (r'.*ed$', 'VERB'),               # past tense 
    (r'.*es$', 'VERB'),               # verb    
    (r'.*\'s$', 'NOUN'),              # possessive nouns
    (r'.*s$', 'NOUN'),                # plural nouns
    (r'\*T?\*?-[0-9]+$', 'X'),        # X
    (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'), # cardinal numbers
    (r'.*', 'NOUN')                   # nouns
]
rule_based_tagger = nltk.RegexpTagger(patterns)


# In[50]:


def Viterbi_rule_based(words, train_bag = train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    for key, word in enumerate(words):
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
        pmax = max(p)
        state_max = rule_based_tagger.tag([word])[0][1]       
        if(pmax==0):
            state_max = rule_based_tagger.tag([word])[0][1] # assign based on rule based tagger
        else:
            if state_max != 'X':
                state_max = T[p.index(pmax)]                
        state.append(state_max)
    return list(zip(words, state))


# In[ ]:





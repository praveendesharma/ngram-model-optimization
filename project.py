# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    response = requests.get(url, timeout=0.5)
    text = response.text
    content = text.split('***')[2]
    content = content.replace('\r\n', '\n')
    
    return content
    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    paragraphs = [p.strip() for p in book_string.split('\n\n') if p.strip()]
     
    tokens = []
    for para in paragraphs:
        tokens.append('\x02')
        tokens.extend(re.findall(r'\w+|[^\w\s]', para))
        tokens.append('\x03')
    
    if not tokens:
        return ['\x02', '\x03']

    if tokens[0] != '\x02':
        tokens.insert(0, '\x02')
    if tokens[-1] != '\x03':
        tokens.append('\x03')
    
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        tokens= np.array(tokens)
        tokens = np.unique(tokens)
        probability=pd.Series(1/len(tokens),index=tokens)
        return probability
    
    def probability(self, words):
        if not words:
            return 0
        
        words_dict = self.mdl.to_dict()
        
        prob = 1
        
        for word in words:
            prob *= words_dict.get(word, 0)
        
        return prob
        
    def sample(self, M):
        words = self.mdl.index
        return ' '.join(np.random.choice(words, size=M, p=self.mdl))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        tokens= np.array(tokens)
        probabilities = pd.Series(tokens).value_counts() / len(tokens)
        return probabilities
    
    def probability(self, words):
        if not words:
            return 0
        
        words_dict = self.mdl.to_dict()
        
        prob = 1
        
        for word in words:
            prob *= words_dict.get(word, 0)
        
        return prob
        
    def sample(self, M):
        words = self.mdl.index
        return ' '.join(np.random.choice(words, size=M, p=self.mdl))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        
        return [tuple(tokens[i:i+self.N]) for i in range(len(tokens) - self.N  + 1)]
        
    def train(self, ngrams):
        # N-Gram counts C(w_1, ..., w_n)
        ngram_col = pd.Series(ngrams)
        n1gram_col = ngram_col.apply(lambda x: (x[:self.N-1]))
        ngram_col_w_counts = ngram_col.value_counts()
        df = pd.DataFrame(ngram_col_w_counts).reset_index().rename(columns={'count':'prob', 'index':'ngram'})


        # (N-1)-Gram counts C(w_1, ..., w_(n-1))

        df['n1gram'] = df['ngram'].apply(lambda x: (x[:self.N-1]))
        


        # Create the conditional probabilities
        n1gram_counts = n1gram_col.value_counts()

        n1gram_dict = n1gram_counts.to_dict()
        df['prob'] = df.apply(lambda row: row['prob'] / n1gram_dict.get(row['n1gram'], 0), axis=1)
    
        # Put it all together
        return df [['ngram','n1gram','prob']]
    
    def probability(self, words):
        total_prob = 1
        
        curr_model = self.prev_mdl
        i = self.N - 2
        while i > 2:
            curr_word = curr_model.create_ngrams(words)[0]
            curr_df = curr_model.mdl.set_index('ngram')
            word_prob = curr_df['prob'].to_dict().get(curr_word, 0)
            total_prob *= word_prob
            curr_model = curr_model.prev_mdl
            i -= 1

        prob_first_word = curr_model.mdl.to_dict().get(words[0], 0)
        total_prob *= prob_first_word
 
        words_grams = self.create_ngrams(words)
        curr_df = self.mdl.set_index('ngram')
        words_prob_dict = curr_df['prob'].to_dict()

        for i in range(len(words_grams)):
            curr_word = words_grams[i]
            word_prob = words_prob_dict.get(curr_word, 0)
            total_prob *= word_prob
        return total_prob
    

    def helper(self, curr_n1gram):
        if self.N - 1 == len(curr_n1gram):
            n1gram_df = self.mdl[self.mdl['n1gram'] == curr_n1gram]
            max_prob = n1gram_df['prob'].max()
            if not max_prob:
                return tuple(list(curr_n1gram) +['\x03'])
            high_p_ngrams = n1gram_df[n1gram_df['prob'] == max_prob]['ngram'].to_list()      
            high_p_ngrams = [x[-1] for x in high_p_ngrams]
            new_token = np.random.choice(high_p_ngrams)
            return new_token
        else:
            return self.prev_mdl.helper(curr_n1gram)         
    
    def sample(self, M):
        # Use a helper function to generate sample tokens of length `length`
        start_char = '\x02'
        end_char = '\x03'
        sentence_list = [start_char]

        for i in range(M-1):
            curr_n1gram = tuple(sentence_list[-(self.N-1):])
            new_token = self.helper(curr_n1gram)
            sentence_list.append(new_token)
        sentence_list.append(end_char)
        return ' '.join(sentence_list)
        # Transform the tokens to strings


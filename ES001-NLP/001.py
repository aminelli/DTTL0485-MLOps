# pip intsall spacy
# python -m spacy download en_core_web_sm
# https://aka.ms/vs/16/release/vc_redist.x64.exe

import spacy
nlp = spacy.load('en_core_web_sm')

introduction_text = ('London is the capital of the UK,'
                     ' with a population of nearly 9 millions people.'
                     ' London is one of the most diverse cities in the world.'
                     ' London has over 100 museums, galleries and exhibitions.'
                     ' London has 40 universities and higher education institutions.'
                     ' London has over 15,500 restaurants, serving Italian, Indian, Thai and Chinese cuisines.'
                     ' London is also one of the world''s capitals of finance, fashion, arts and entertainment.')


introduction_doc = nlp(introduction_text)

print("#Sentences =====================================================")
sentences = list(introduction_doc.sents)
print(len(sentences))
for sentence in sentences:
     print (sentence)
     
#
     

# Extract tokens for the given doc
print("#Tokens ========================================================")
for token in introduction_doc:
     print (token, token.idx)
for token in introduction_doc:
     if not token.is_stop:
         print (token)

print("#Lemmatization =================================================")
#Lemmatization is the process of reducing inflected forms of a
#word while still ensuring that the reduced form belongs to the language.
#This reduced form or root word is called a lemma.
for token in introduction_doc:
     print (token, token.lemma_)


print("#Word Frequency =================================================")
from collections import Counter
# Remove stop words and punctuation symbols
words = [token.text for token in introduction_doc
          if not token.is_stop and not token.is_punct]
word_freq = Counter(words)
# 5 commonly occurring words with their frequencies
common_words = word_freq.most_common(5)
print (common_words)

print("#Unique words ===================================================")
unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
print (unique_words)

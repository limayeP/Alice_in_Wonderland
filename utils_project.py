from urllib import request
import collections
from collections import OrderedDict
import contractions
import re
import nltk
from nltk import word_tokenize, sent_tokenize
import string
from nltk.corpus import stopwords
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from nltk.probability import FreqDist
import pronouncing
from nltk.corpus import wordnet


def read_gutenberg(url):
    # Input: text to be analyzed
    response = request.urlopen(url)
    # Read as one long string 
    raw = response.read().decode('utf-8-sig')
    # To find the beginning and end of the text
    start = raw.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = raw.rfind("*** END OF THE PROJECT GUTENBERG EBOOK")
    raw = raw[start:end]
    return raw

def chapter_titles(raw):
    nchar = 7
    nlr = 5
    s = []
    titles = []
    chapters = []
    c = raw.split('CHAPTER ')[1:]
    for a in c:
        if len(re.findall(r'\n', a)) > nlr:
            s.append(a)
        else:
            titles.append(a[nchar:].strip())
    for ix, t in enumerate(titles):
        chapters.append(s[ix].split(t)[1].strip().replace('THE END', ''))
    return chapters, titles

def cleanup_special_characters(s):
    q = [ord(c) for c in s]
    qq = [i for i in q if i < 256]
    res = ''.join(chr(val) for val in qq)
    return res

def odd_ones(chapter):
    chapter = chapter.replace("Beau—ootiful", 'Beautiful')
    chapter = chapter.replace("beauti—FUL", 'beautiful')
    chapter = chapter.replace("Soo—oop", 'Soup')
    chapter = chapter.replace("e—e—evening", 'evening')
    chapter  = chapter.replace("Dinah'll", "Dinah")
    chapter  = chapter.replace("Alice's", "Alice")
    chapter = chapter.replace("ennyworth", "Pennyworth")
    chapter = re.sub(r"\bp\b", "", chapter)

    return chapter

def tokenize(chapter):
    """
    Input: chapter from text
    Output(1): text compiling all the chapters
    Output(2): wordlist from all the chapters
    """
    chapter = chapter.replace("—", ' ')
    chapter = chapter.replace("_", ' ')
    expanded_words = []
    for word in chapter.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    z = cleanup_special_characters(expanded_text)
    t = "".join([i for i in z if i not in string.punctuation])
    tokens = word_tokenize(t)
     # Take care of Miss and miss
    word_list = ["Miss"]
    t_words = [word for word in tokens if word not in word_list]
    tt_words = [word.replace("Duchesss", "Duchess") for word in t_words]
    # Convert to lower case
    # small_words = [w.lower() for w in t_words]
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tt_words if w.lower() not in stop_words]
   # final_words = [w for w in small_words if not w in stop_words]
    processed_words = []
    for w in filtered_words:
        if len(w) != 1:
            processed_words.append(w)
    
    text = ' '.join(processed_words)
    return processed_words, text

def check(text):
    # Input text or wordlist
    # Check for items
    # Output count of items 
    counts = collections.Counter(text)
    counts_sorted = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    return  counts, counts_sorted

def word_distribution(words):
    # Calculate the percentage of words in each chapter
    # Input chapter words as a list of lists
    s = []
    for t in words:
        s.append(len(t))
    ss = [(k/sum(s))*100 for k in s]
    return ss

def flatten_list(wordlist):
    # Flatten the wordlist
    regular_list = wordlist
    flat_list = [item for sublist in regular_list for item in sublist]
    return flat_list
   
# Count them up by length.
def count_by_wordlength(words):
    # input wordlist
    e  = {}
    for word in words:
        wordLength = len(word)
        if wordLength in e:
            e[wordLength] += 1
        else:
            e[wordLength] = 1
    for i in sorted(e):
        e[i]/= len(words)
        e[i]*= 100
    return e

def longest_word(words):
    # # To find the longest word in list
    e = {}
    for word in words:
        e[word] = len(word)
    return  max(e, key=e.get) 

def shortest_word(words):
    # # To find the shortest word (not one word)in list
    # Input: List of words
    # Output: dictionary of words that are of length 2 and lesser
    e = {}
    for word in words:
        if len(word) < 3:
            e[word] = len(word)
    return e

def sort_letters(s):
     # sorted splits word into sorted character list
    return''.join(sorted(s))

def anagram(words):
    # get anagram
    c  = {}
    for word in set(words):
        key = sort_letters(word)
        if key in c:
            c[key] = c[key] + ' & ' + word
        else:
            c[key] = word
    anagram  = {}
    for k, v in c.items():
        if '&' in v:
            anagram[k] = {}
            anagram[k]['value'] = v
            anagram[k]['count'] = len(re.findall('&', v)) + 1
    return anagram

def anagram_count(r, n):
    # r is a dictionary of anagrams
    # count the words with a n anagrams
    a = []
    for k, v in r.items():
        if v['count'] == n:
            a.append(v['value'])
    return a

def initial_yToc(words):
    # for words starting with y, change the starting y to c 
    b = []
    for word in words:
        # change initial y’s to a ‘c
        word = re.sub(r'^y',r'c',word)
        b.append(word)
    return b

def begin_v(g):
    # input g is a list containing words not begining with y
    # Find words begining with vowels
    begin = []
    for word in g:
        if word[0] in 'aeiou':
            begin.append(word)
    return begin

def end_v(g):
    # input g is a list containing words not begining with y
    # Find words ending with vowels
    end = []
    for word in g:
        if word[-1] in 'aeiou':
            end.append(word)
    return end

def is_palindrome(word):
    ispalindrome = bool(0)
    if len(word) > 1 and word == word[::-1]:
        ispalindrome = bool(1)
    return ispalindrome

def repeated_three(wordlist):
    # a word with a letter repeated three times in
    regex = re.compile(r'(.)\1\1')
    matches = [word for word in wordlist if re.search(regex, word)]
    return matches
     
def cv(word):
    # get cv paterns of words
    word = re.sub(r'^y',r'c', word)
    word = re.sub(r'[^aeiouy]', r'c', word)
    return(re.sub(r'[aeiouy]', r'v', word))

def is_univocalic(w):
    v = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0, 'y': 0}
    e = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0, 'y': 0}
    for c in w:
        if c in ['a', 'e', 'i', 'o', 'u', 'y']:
            e[c] = 1
            v[c] += 1
    u = 1
    if sum(e.values()) > 1:
        u = 0
    return u, v, sum(v.values())

def univocalic_words(words, vowel):
    # vowel = any vowel to find
    uw = []
    for w in words:
        if is_univocalic(w)[0]:
            if vowel in w:
                uw.append(w)
    return uw

# To get concordance of words
def make_regex(name, width):
    name_CamelCase = re.sub(r'[. ]', r'', name) # Remove periods, spaces
    name_escape = name.replace('.', '\.') # Escape periods in regex
    return "re_" + name_CamelCase + " = r'.{" + str(width) + "}" + name_escape + ".{" + str(width) + "}'"

# Use the Word Corpus to find unsual or mis-spelt words
def unusual_words(words):
    unusual = []
    # Provide word list
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    for word in words:
        if word not in english_vocab:
            unusual.append(word)
    return sorted(set(unusual))

# Make a rhyming dictionary
def rhyming_dict(words):
    """
    Input a wordlist
    Output  dict for keys with values
    """
    rhyming_prim = {}
    rhyming_d = {}
    for k in words:
        fh = pronouncing.rhymes(k)
        rhyming_prim[k] = set(fh).intersection(words)
        if rhyming_prim[k] != set():
            rhyming_d[k] = rhyming_prim[k]
    return rhyming_d

def dict_num_values(word_dict):
    # Input: Dict where values are wordlists
    # Output: Dict where values is the number of values
    length_dict = {key: len(value) for key, value in word_dict.items()}
    return length_dict

def max_num_values(length_dict):
    # Input: Dict where values are number of values
    # Output: Largest number of values and corresponding key in that prder
    vv = 0
    for k, v in length_dict.items():
        if v > vv:
            vv = v
            kk = k
    return vv, kk

def pronounce_dict(word_list):
    # Input a wordlist
    # Output dict with pronounciation of wordlist
    # import pronouncing
    a = {}
    b = {}
    for w in word_list:
        ch = pronouncing.phones_for_word(w)
        if ch:
            a[w] = ch[0]
        else:
            b[w] = ch
    return a

def keys_same_values(files):
    names = set(files.values())
    d = {}
    for n in names:
        d[n] = [k for k in files.keys() if files[k] == n]
    return d
        
def homophones(flipped_dict):
    # Input dict where keys and values are flipped
    # Get homophones: two different words with same pronounciation
    b = {}
    for k, v in flipped_dict.items():
        if len(v) > 1:
            b[k] = v
    return b

def get_nouns(alice_pos):
    # Input wordlist with POS tags
    # Output nouns
    only_nouns = []
    nouns = [ 'NN', 'NNP', 'NNPS', 'NNS']
    for token in alice_pos:
        for symbol in nouns:
            if symbol == token[1]:
                only_nouns.append(token[0])
    return only_nouns

# Get only adjectives
def get_adj(alice_pos):
    # Input wordlist with POS tags
    # Output adjectives    
    only_adj = []
    adj = ['JJ', 'JJR', 'JJS']
    for token in alice_pos:
        for symbol in adj:
            if symbol == token[1]:
                only_adj.append(token[0])
    return only_adj

# Get only the verbs
def get_verb(alice_pos):
    # Input wordlist with POS tags
    # Output verb    
    only_verb = []
    verb = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    for token in alice_pos:
        for symbol in verb:
            if symbol == token[1]:
                only_verb.append(token[0])
    return only_verb

def sort_listoflists(ll):
    # Sort list of lists by the first element
    ll.sort(key=lambda x: x[0])
    return ll

# Get nouns into a list from a dictionary
def get_nouns_from_dict(dt):
    # Provide dict with words and wordtags
    # Returns list of nouns
    n = []
    for k, v in dt.items():
        if (dt[k]) == 'noun':
            n.append(k)
    return n

def word_count(words):
    """
    count number sorted occurances of words
    """
    e = {}
    words = set(words)
    for word in words:
        wordLength = len(word)
        if wordLength in e:
            e[wordLength].append(word)
        else:
            e[wordLength] = [word]
    return e

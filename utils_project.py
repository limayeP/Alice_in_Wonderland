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
    """
    Input: text to be analyzed
    Output: string of raw text 
    """
    response = request.urlopen(url)
    # Read as one long string 
    raw = response.read().decode('utf-8-sig')
    # To find the beginning and end of the text
    start = raw.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = raw.rfind("*** END OF THE PROJECT GUTENBERG EBOOK")
    raw = raw[start:end]
    return raw

def chapter_titles(raw):
    """
    Input: raw text
    Output(1): chapters = list of text of each chapter
    Output(2): titles = list of title of each chapter
    Output(3): sum(totl_l) =  total number of characters in the entire text
    Output(4): sum(tottl_w) = total number of words before cleaning in the entire text
    """
    nchar = 7
    nlr = 5
    
    totl_l = []
    totl_w = []
    chapter_text = []
    titles = []
    chapters = []
    c = raw.split('CHAPTER ')[1:]
    for a in c:
        if len(re.findall(r'\n', a)) > nlr:
            chapter_text.append(a)
        else:
            titles.append(a[nchar:].strip())
    for ix, t in enumerate(titles):
        chapters.append(chapter_text[ix].split(t)[1].strip().replace('THE END', ''))
    for c in chapters:
        o = len(c)
        p = word_tokenize(c)
        r = len(p)
        totl_l.append(o)
        totl_w.append(r)
    sum_chars = sum(totl_l)
    sum_words = sum(totl_w)
    return {"chapters": chapters, "titles": titles,
            "sum_chars": sum_chars, "sum_words": sum_words}

def cleanup_special_character(text):
    """
    Input: string of text to be cleaned
    Output(1): chapters = list of chapter text of each chapter
    Output(2): titles = list of title of each chapter
    Output(3): totl_l = list of number of characters in each chapter
    Output(4): tottl_w = list of number of words in each chapter
    """
    q = [ord(c) for c in text]
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
    z = cleanup_special_character(expanded_text)
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

def flatten_list(wordlist):
    # Flatten the wordlist
    regular_list = wordlist
    flat_list = [item for sublist in regular_list for item in sublist]
    return flat_list
   
def get_cleaned_Alice_text(chapters):
    """
    Purpose: To clean the text
    Input: chapters = list of text of each chapter
    Output(1): text = text compiling all the chapters
    Output(2): wordlist = wordlist list of lists of words from all the chapters
    Output(3): Words= flattened wordlist of all the words from all the chapters
    Output(4): num_of_words = total number of words after cleaning
    Output(5): num_of_unique_words = total number of unique words after cleaning
    Output(6): frequency distribution of words
    """
    text = []
    wordlist = []
    for c in chapters:
        c = odd_ones(c)
        wl, txt = tokenize(c)
        wordlist.append(wl)
        text.append(txt)

    # list of lists to list
    words_dirty = flatten_list(wordlist)
    num_of_words = len(words_dirty)
    num_of_unique_words = len(set(words_dirty))
    fdist_words_dirty = FreqDist(words_dirty)
    return {"text":text, "wordlist": wordlist, "words_dirty": words_dirty, "num_of_words" :num_of_words, "num_of_unique_words": num_of_unique_words,"fdist_words_dirty":
 fdist_words_dirty}

def freq_dist_of_n_most_common_words(num_of_most_common_words, fdist_words_dirty):
    """
    Purpose: To clean the text
    Input: num_of_most_common_words = number of most common words
    Output(1): freq_dist = list of tuples containing word and its frequency
    Output(2): frequency distribution plot
    """
    freq_dist = fdist_words_dirty.most_common(num_of_most_common_words)
    # Frequency Distribution Plot
    fdist_words_dirty.plot(num_of_most_common_words,cumulative=False)
    return freq_dist, plt.show()

def num_Alice_mentions(chapters):
    """
    Purpose: To clean the text
    Input: chapters = list of text of each chapter
    Output(1):dict of ket = chapter number, value = freq of "alice/Alice"
    Output(2):plot of the number of times ' Alice/alice' occurs in each chapter

    """
    alice_count = []
    for c in chapters[0:]:
        alice_count.append(c.count("Alice"))
    # combine the chapter titles (keys) and “Alice” counts
    # (values) and convert them to a dictionary.
    mydict = dict(zip(range(1, len(chapters)+1), alice_count))
    plt.bar(range(1, len(chapters)+1), alice_count, edgecolor='black')
    plt.title('Distribution of "Alice" across chapters')
    plt.xlabel('Chapters')
    plt.ylabel('Alice count')
    plt.show()
    return mydict

def check(text):
    # Input text or wordlist
    # Check for items
    # Output count of items 
    counts = collections.Counter(text)
    counts_sorted = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    return  counts, counts_sorted

# Count them up by length.
def count_by_wordlength(words_dirty):
    """
    Input: Word list containing both upper and lower case letters
    Output(1):e = dict of key = length of word, value = Percentage of words of that length
    Output(2): words = list of words all lower-case
    Output(3): countsa_sorted = word count sorted
    Output(4): Plot of the distribution of the words by length
    """
    words = []
    for w in words_dirty:
        a = w.lower()
        words.append(a)
    countsa, countsa_sorted = check(words)
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
    for keys,values in e.items():
        x = e.keys()
        y = e.values()
        plt.bar(e.keys(), e.values(), edgecolor='black')
        plt.title('Fig2 Distribution of words by length')
        plt.xlabel('Word Length')
        plt.ylabel('Percentage of words')
    plt.show()
    return {"word_dict": e, "words": words, "count_sorted": countsa_sorted}

def lexical_diversity(word_list):
    """
    Goal: Lexical Diversity in wordlist
    Input : wordlist
    Output: Lexical Diversity
    """
    lexical_diversity_words = len(word_list)/len(set(word_list))
    return lexical_diversity_words

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

def sort_letters(chapter_text):
     # sorted splits word into sorted character list
    return''.join(sorted(chapter_text))

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

def repeated_three(wordlist):
    # a word with a letter repeated three times in
    regex = re.compile(r'(.)\1\1')
    matches = [word for word in wordlist if re.search(regex, word)]
    return matches
 
def is_palindrome(word):
    ispalindrome = bool(0)
    if len(word) > 1 and word == word[::-1]:
        ispalindrome = bool(1)
    return ispalindrome

def palindromes(words):
    """
    Goal: To find Palindromes
    Input: wordlist from cleaned text
    Output: list of palindromes
    """
    palin = []
    words = list(words)
    for w in set(words):
        if is_palindrome(w):
            palin.append(w)
    return palin

def initial_yToc(words):
    # for words starting with y, change the starting y to c
    b = []
    for word in words:
        # change initial y’s to a ‘c
        word = re.sub(r'^y',r'c',word)
        b.append(word)
    return b

def begin_v(words):
    """
    Goal: Find words begining with vowels
    Input: wordlist
    Output: list of words beginning with vowels
    """
    g = initial_yToc(words)
    begin = []
    for word in g:
        if word[0] in 'aeiou':
            begin.append(word)
    return begin

def end_v(words):
    """
    Goal: Find words ending with vowels
    Input: wordlist
    Output: list of words ending with vowels
    """
    g = initial_yToc(words)
    end = []
    for word in g:
        if word[-1] in 'aeiou':
            end.append(word)
    return end

def cv(word):
    """
    Goal: get cv patterns of words
    Input: word
    Output: word in the form of a cv pattern
    """
    word = re.sub(r'^y',r'c', word)
    word = re.sub(r'[^aeiouy]', r'c', word)
    return(re.sub(r'[aeiouy]', r'v', word))

def dict_cv(words):
    """
    Goal: Create a dictionary with key = cv-pattern
    Input: wordlist
    Output(1): dict with key = cv-pattern  and value = list of words having that pattern.
    Output(2): total number of patterns representing words in the text
    """
    d = {}
    for word in set(words):
        key = cv(word)
        if key in d:
            d[key].append(word)
        else:   
             d[key] = [word]
    key_length = len(d.keys())
    return d, key_length

def patterns_only_vowels(d):
    """
    Input: dictionary with key = cv-pattern and value = list of words having that pattern
    Output: list of patterns with only vowels
    """
    reg_pattern1 = re.compile(r'\bv+v$')
    matching_values1 = [v for v in d if re.match(reg_pattern1, v)]
    return matching_values1

def patterns_v_c(d):
    """
    Goal: Find words with v followed by c pattern
    Input: dictionary with key = cv-pattern and value = list of words having that pattern
    Output: list of patterns  with v followed by c
    """
    reg_pattern2 = re.compile(r'^(vc)+$')
    matching_values2 = [v for v in d if re.match(reg_pattern2, v)]
    return matching_values2

def patterns_c_v(d):
    """
    Goal: Find words with c followed by v pattern
    Input: dictionary with key = cv-pattern and value = list of words having that pattern
    Output: list of patterns with c followed by v
    """
    reg_pattern3 = re.compile(r'^(cv)+$')
    matching_values3 = [v for v in d if re.match(reg_pattern3, v)]
    return matching_values3

def sort_values_dict_into_array(d):
    """
    Goal: Sort values of dictionary d in to an array and take logarithm
    Input: Dictionary
    Output(1): logarithm of the counts of the array of values of dcitionary
    Output(2): histogram of the logarithm of counts of the array of the values of a dictionary
    """
    counts = sorted(np.array([len(value) for value in d.values()]))
    logCounts = np.array([math.log(len(value)) for value in d.values()])
    plt.hist(logCounts, edgecolor='black')
    plt.title('Fig3 Histogram of cv-pattern lengths')
    plt.xlabel('Logarithm of length of cv patterns')
    plt.ylabel('Number of cv patterns')
    return logCounts, plt.show()

def dict_with_keys_length_values(d):
    """
    Goal: dictionary with keys of the len(d[k])
    Input: d = dictionary where 
    Output(1): ordered dictionary with keys of the len(d[k])
            Each value is a list of keys of ‘d.’
    Output(2): Number of  patterns are represented by one word.
    """
    dlen = {}
    for k, v in d.items():
        key = len(v)
        if key in dlen:
            dlen[key].append(k)
        else:
            dlen[key] = [k]
    dict1 = OrderedDict(sorted(dlen.items()))
    l = len(dict1[1])
    return  dict1, l

def Save_dict(d, filepath):
    """
    Goal: Save a dict
    Input(1): dictionary to be saved
    Input(2): string of the filepath with the name to save the file as
    """
    with open(filepath, "w") as f:
        for length in sorted(d.keys()):
            f.write(f'{length}, {d[length]}\n\n')
            
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

def univocalic_words_single_vowel(words, plot=False):
    """
    Goal: Univocalic words containing a single type of vowel
    Input: wordlist
    Output(1): percentage of Univocalic words
    Output(2): sum of Univocalic words
    Output(3): Plot of the distributon of univocalic words
    """
    vowel = ['a', 'e', 'i', 'o', 'u', 'y']
    g = {}
    for v in vowel:
        g[v] = len(set(univocalic_words(words, v)))
        univocalic = []
        for gi in g.values():
            a =round((gi *100)/sum(g.values()),2)
            univocalic.append(a)
    if plot:
        plt.bar(g.keys(), univocalic, edgecolor='black')
        plt.title('Fig4 Distribution of Univocalic words in the text')
        plt.xlabel('Vowels')
        plt.ylabel('Percentage of words')
    plt.show()
    return a

def plot_univocalic_word_distribution(words):
    vowel = ['a', 'e', 'i', 'o', 'u', 'y']
    g = {}
    for v in vowel:
        g[v] = len(set(univocalic_words(words, v)))
    plt.bar(g.keys(), univocalic, edgecolor='black')
    plt.title('Fig4 Distribution of Univocalic words in the text')
    plt.xlabel('Vowels')
    plt.ylabel('Percentage of words')
    plt.show()
    
def hepax_legomena_entire_text(words):
    """
    Goal: Hepax Legomena :Checking for words that occur only once in the entire text
    Input: list of words
    Output: count of words that occur only once in the wordlist, words themselves that ocu only once
    """
    fdist = FreqDist(words)
    hapax_legomenas = fdist.hapaxes()
    hapax_legomena_counts = len(hapax_legomenas)
    return hapax_legomena_counts, hapax_legomenas

def hepax_legomena_by_chapter(wordlist_by_chapter,chapters, plot=False):
    """
    Input: wordlist by chapter, chapters, to plot or not)
    Output(1):list of the number of hepax legomenas in each chapter
    Output(2):Percetage of hepax legomeas in each chapters
    Output(3): plot of Percentage of Hapax Legomeas in each chapter
    """
    # Get the frequency distribution of words in each chapter
    fdist1 = [FreqDist(w) for w in wordlist_by_chapter]
    # Number of Hepax Legomena in each chapter
    chap_hepax = []
    for f in fdist1:
        hapax_legomenas_1 = len(f.hapaxes())
        chap_hepax.append(hapax_legomenas_1)
    tot_word_each_chapter= []
    for w in wordlist_by_chapter:
        tot_word_each_chapter.append(len(w))
    per_hepax = [round(int(i)/int(j)*100, 2) for i, j in zip(chap_hepax,tot_word_each_chapter )]
    if plot:
        plt.bar(range(1, len(chapters)+1),per_hepax ,edgecolor='black')
        plt.title('Fig5 Distribution of Hapax Legomena per chapter')
        plt.xlabel('Chapters')
        plt.ylabel('Percentage of Hapax Legonema')
    plt.show()

    return chap_hepax, per_hepax

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

def get_longest_and_shortest_rhyming_words(words):
    """
    Input: wordlist
    Output(1): Get a dict of longest and shortest rhyming words and their lengths
    Output(2): dict of  rhyming words
    Example of Output(2)'fast': {'asked', 'last', 'passed', 'past'},
    """
    rhy  = rhyming_dict(words)
    dict_rhy = {}
    # Longest rhyming words
    longest_k =  max(len(x) for x in rhy)
    for x in rhy:
        if len(x) == longest_k:
            dict_rhy[x] = len(x)
    # Shortest rhyming words
    shortest_k = min(len(y) for y in rhy)
    for y in rhy:
        if len(y) == shortest_k:
             dict_rhy[y] = len(y)
    return dict_rhy, rhy

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

def dict_number_of_rhyming_words(rhy):
    """
    Input: dict of rhyming words. Example 'fast': {'asked', 'last', 'passed', 'past'}
    Output(1): Tuple of number of rhyming words, key. Example (16, 'day')
    Output(2): word with largest number of rhyming words
    """
    # Dict where values are the number of rhyming words
    len_dict = dict_num_values(rhy)
    # max value and corresponding key
    max_v, max_k = max_num_values(len_dict)
    max_v, max_k
    # Key with largest number of rhyming words
    rhy[max_k]
    return (max_v, max_k), rhy[max_k]

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
        
def homophones(flipped_dict):
    # Input dict where keys and values are flipped
    # Get homophones: two different words with same pronounciation
    b = {}
    for k, v in flipped_dict.items():
        if len(v) > 1:
            b[k] = v
    return b

def homophone_dict(words):
    """
    Input: list of words
    Output: dict of homophones
    """
    # Build a reference word pronounciation dictionary
    pro = pronounce_dict(words) 
    # Flip keys and values
    pro_dict = keys_same_values(pro)
    homop = homophones(pro_dict)
    return homop

def find_collocations(words):
    """
    Input: wordlist
    Output: the 10 most common bigrams, trigrams, quadrigrams
    """
    finder = {}
    finder["two"] = nltk.collocations.BigramCollocationFinder.from_words(words)
    finder["three"] = nltk.collocations.TrigramCollocationFinder.from_words(words)
    finder["four"] = nltk.collocations.QuadgramCollocationFinder.from_words(words)
    for g in finder:
        plt.figure(dpi=100, figsize=(7, 5))
        plt.title(f"Fig6_{g}.ngrams")
        plt.xlabel(f"{g}.ngram")
        plt.ylabel("frequency")
        plt.tick_params(axis='both', which='major', labelsize=10)
        rcParams.update({'figure.autolayout': True})
        plt.tight_layout()
        finder[g].ngram_fd.plot(10)


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

def pos_tagging(words):
    """
    Input: Use Word list which contains uppercase and lowecase words
           nltk.help.upenn_tagset()
    Output: dict with lists of nouns, adjectives and verbs
    """
    alice_pos = nltk.pos_tag(words)
    alice_pos = list(map(list, alice_pos))
    # Get nouns
    n = get_nouns(alice_pos)
    # Get only adjectives
    adj = get_adj(alice_pos)
    # Get only verbs
    verb = get_verb(alice_pos)
    return {"nouns": n, "adjectives": adj, "verbs": verb}

def plot_most_common_nouns_adjs_verbs(dict_pos_tags, number=10):
    binder={}
    binder["freq nouns"] = FreqDist(dict_pos_tags["nouns"])
    binder["freq adjectives"] = FreqDist(dict_pos_tags["adjectives"])
    binder["freq verbs"] = FreqDist(dict_pos_tags["verbs"])
    for g in binder:
        plt.figure(dpi=100, figsize=(7, 5))
        plt.title(f"Fig7_{g}")
        plt.xlabel(f"{g}")
        plt.ylabel("frequency")
        plt.tick_params(axis='both', which='major', labelsize=10)
        rcParams.update({'figure.autolayout': True})
        plt.tight_layout()
        binder[g].plot(number, cumulative=False)

def get_most_common_main_characters(noun_list):
    """
    Input: list of nouns
    Output: list of main characters in the story
    """
    freqn = FreqDist(noun_list)
    # 25 most common nouns
    common_noun_list = []
    nq = freqn.most_common(25)
    for a in  nq:
        common_noun_list.append((a[0]))
    main_characters = []
    for b in common_noun_list:
        if b[0].isupper():
            main_characters.append(b)
    return main_characters

def dict_count_of_main_characters(chapter_list, main_characters):
    """
    Input(1): list of text chapters
    Input(2): list of main characters
    Output: dict of key = main character, value = number of occurances
    """
    grinder = {}
    for k in main_characters:
        grinder[k] = []
        for c in chapter_list[0:]:
            grinder[k].append(c.count(k))
    return grinder

def plot_main_characters(main_character_counts, chapter_list):
    """
    Input: dict of character counts of main characters
    Output: Plot of the frequency of main characters across chapters
    """
    rcParams.update({'figure.autolayout': True})
    for g in main_character_counts:
        plt.figure(dpi=100, figsize=(7, 5))
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        x = range(1, len(chapter_list)+1)
        plt.bar(x, main_character_counts[g], edgecolor='black')
        plt.title(f"frequency of occurrence")
        plt.xlabel("chapter number")
        plt.xticks(x)
        plt.ylabel(f"occurrence of {g}")
        plt.grid("on")
    plt.show()

def sort_listoflists(ll):
    # Sort list of lists by the first element
    ll.sort(key=lambda x: x[0])
    return ll

def keys_same_values(files):
    names = set(files.values())
    d = {}
    for n in names:
        d[n] = [k for k in files.keys() if files[k] == n]
    return d

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

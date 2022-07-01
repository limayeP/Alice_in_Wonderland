 #!/usr/bin/python

from utils_project import *

url = "https://www.gutenberg.org/files/11/11-0.txt"
raw = read_gutenberg(url)

# Split chapters
chapters, titles = chapter_titles(raw)

# Initial view
# Number of chapters
print("Number of chapters: ", len(chapters))

# Total length of all the chapters
totl_l = []
totl_w = []
for c in chapters:
    o = len(c)
    p = word_tokenize(c)
    r = len(p)
    totl_l.append(o)
    totl_w.append(r)
    
print("The total number of characters in all the chapters are" , sum(totl_l))
print("The total number of words before cleaning is the text are" , sum(totl_w))

#############################################################
# Cleaning
############################################################
text = []
wordlist = []
for c in chapters:
    c = odd_ones(c)
    wl, txt = tokenize(c)
    wordlist.append(wl)
    text.append(txt)

# list of lists to list
Words = flatten_list(wordlist)

print("The total number of words after cleaning are" , len(Words))

print("The total number of unique words after cleaning are" , len(set(Words)))

# 10 most commmon words
fdist_Words = FreqDist(Words)
fdist_Words.most_common(10)
# Frequency Distribution Plot
fdist_Words.plot(20,cumulative=False)
plt.show()

# Number of times Alice has been mentioned in each chapter
alice_count = []
for c in chapters[0:]:
    alice_count.append(c.count("Alice"))
# combine the chapter titles (keys) and “Alice” counts
# (values) and convert them to a dictionary.
mydict = dict(zip(range(1, len(chapters)+1), alice_count))
print(mydict)

plt.bar(range(1, len(chapters)+1), alice_count, edgecolor='black')
plt.title('Distribution of "Alice" across chapters')
plt.xlabel('Chapters')
plt.ylabel('Alice count')
plt.show()
###############################################################
# Word Analysis
###############################################################

# Percent word count per chapter
percent_length = word_distribution(wordlist)
plt.bar(range(1, len(percent_length)+1), percent_length, edgecolor='black')
plt.title('Fig1 Percent distribution  word count across chapters')
plt.xlabel('Chapters')
plt.ylabel('percent word count')
plt.show()


# Convert Words to lower case
words = []
for w in Words:
    a = w.lower()
    words.append(a)
    
# Check the words
countsa, countsa_sorted = check(words)

print("The total number of words after cleaning and case insensitive are" , len(words))

print("The total number of unique words after cleaning and case insensitive are" , len(set(words)))


# Lexical diversity in words and Words
# words has words (only lower case alphabets)
# Words has words (with upper and lower case alphabets)
lexical_diversity_words = len(words)/len(set(words))
lexical_diversity_words
print("Each word(all lowercase) is used ",lexical_diversity_words , "times on average")

lexical_diversity_Words = len(Words)/len(set(Words))
print("Each word(upper and lowercase) is used ",lexical_diversity_Words , "times on average")


# Count number of words  by their length.
e = count_by_wordlength(words)
for keys,values in e.items():
    x = e.keys()
    y = e.values()
plt.bar(e.keys(), e.values(), edgecolor='black')
plt.title('Fig2 Distribution of words by length')
plt.xlabel('Word Length')
plt.ylabel('Percentage of words')
plt.show()

# Length of longest word
longestWord = longest_word(words)
print("The longestword: ",longestWord, len(longestWord))
shortestWord = shortest_word(words)
print("The shortestword: ",shortestWord, len(shortestWord))

# a word with a letter repeated three times in
i = repeated_three(words)
print("Number of repeated three letter words: ", i)

####################################
# Get anagrams
####################################
y = anagram(words)
# count the words with a n anagrams
print("Number of anagrams: ", len(y))
z = anagram_count(y, 2)
print("Number of anagrams consisting of  2 words: ", len(z))
x = anagram_count(y, 3)
print("Number of anagrams consisting of  3 words: ", len(x))

###########################################
# Find palindromes
############################################
palin = []
for w in set(words):
    if is_palindrome(w):
        palin.append(w)
        # Given a word list picks palindrome from it
        print("Palindromes are:", (w))
print("There are", len(palin),"palindromes in this text")


# Change words begining with 'y' to 'c'
g = initial_yToc(words)
# Find words begining with vowels
begining_with_vowels = begin_v(g)
print("Number of words begining with vowels: ", len(begining_with_vowels))

# Find words ending with vowels
ending_with_vowels = end_v(g)
print("Number of ending with vowels: ", len(ending_with_vowels))
#########################################################
# Create a dictionary with key = cv-pattern
########################################################
# value = list of words having that pattern.
d = {}
for word in set(words):
    key = cv(word)
    if key in d:
        d[key].append(word)
    else:   
         d[key] = [word]

# Total number of patterns representing words in the text
len(d.keys())
        
# Pick up words with only vowels
reg_pattern1 = re.compile(r'\bv+v$')
matching_values1 = [v for v in d if re.match(reg_pattern1, v)]
matching_values1
d['vvv']

# Find words with v followed by c pattern
reg_pattern2 = re.compile(r'^(vc)+$')
matching_values2 = [v for v in d if re.match(reg_pattern2, v)]
matching_values2
d['vcvc']
d['vcvcvcvc']
d['vcvcvc']
d['vc']
                          
# Find words with c followed by v pattern
reg_pattern3 = re.compile(r'^(cv)+$')
matching_values3 = [v for v in d if re.match(reg_pattern3, v)]
matching_values3

# Sort values of dictionary d in to an array and take logarithm
counts = sorted(np.array([len(value) for value in d.values()]))
logCounts = np.array([math.log(len(value)) for value in d.values()])
plt.hist(logCounts, edgecolor='black')
plt.title('Fig3 Histogram of cv-pattern lengths')
plt.xlabel('Logarithm of length of cv patterns')
plt.ylabel('Number of cv patterns')
plt.show()

# New dictionary with keys of the len(d[k]).
# Each value is a list of keys of ‘d.’
dlen = {}
for k, v in d.items():
    key = len(v)
    if key in dlen:
        dlen[key].append(k)
    else:
        dlen[key] = [k]

dict1 = OrderedDict(sorted(dlen.items()))
# How many many patterns are represented by one word.
len(dict1[1])
# with open("/home/plimaye/Documents/CCSU/DATA_531/project_DATA531/alice_cv.txt", "w") as f:
#     for length in sorted(dlen.keys()):
#         f.write(f'{length}, {dlen[length]}\n\n')

#############################################################
# Univocalic words containing a single type of vowel
##############################################################
vowel = ['a', 'e', 'i', 'o', 'u', 'y']
g = {}
for v in vowel:
    g[v] = len(set(univocalic_words(words, v)))
univocalic = []
for gi in g.values():
    a =round((gi *100)/sum(g.values()), 2)
    univocalic.append(a)

print("A total of", sum(g.values())," are univocalic.")
print("Univocalic words comprise", round((sum(g.values())/len(words))*100,2) , "% of the total number of words.")

plt.bar(g.keys(), univocalic, edgecolor='black')
plt.title('Fig4 Distribution of Univocalic words in the text')
plt.xlabel('Vowels')
plt.ylabel('Percentage of words')
plt.show()

vowel = ['a', 'e', 'i', 'o', 'u', 'y']
h = {}
for v in vowel:
    h[v] = set(univocalic_words(words, v))
h
########################################################
# Hepax Legomena
# Checking for words that occur only once in the entire text
# fdist defined on line 47 as freqDist(words)
###########################################################
fdist = FreqDist(words)
hapax_legomenas = fdist.hapaxes()
hapax_legomena_counts = len(hapax_legomenas)
hapax_legomena_counts

# Checking for Hepax Legomena chapter by chapter

# Get the frequency distribution of words in each chapter
fdist1 = [FreqDist(w) for w in wordlist]

# Number of Hepax Legomena in each chapter
chap_hepax = []
for f in fdist1:
    hapax_legomenas_1 = len(f.hapaxes())
    chap_hepax.append(hapax_legomenas_1)
chap_hepax

# Number of words in each chapter
each_hepax = []
for w in wordlist:
    each_hepax.append(len(w))
each_hepax

# Percent of Hepax Legomena per chapter
per_hepax = [round(int(i)/int(j)*100, 2) for i, j in zip(chap_hepax, each_hepax)]

plt.bar(range(1, len(chapters)+1),per_hepax ,edgecolor='black')
plt.title('Fig5 Distribution of Hapax Legomena per chapter')
plt.xlabel('Chapters')
plt.ylabel('Percentage of Hapax Legonema')
plt.show()
#######################################################
# Making a rhyming dictionary
rhy  = rhyming_dict(words)

#Longest rhyming words
longest_k =  max(len(x) for x in rhy)
longest_k
for x in rhy:
    if len(x) == longest_k:
        print(x)
rhy['disappointment']
rhy['multiplication']

# Dict where values are the number of rhyming words
len_dict = dict_num_values(rhy)
# max value and corresponding key
max_v, max_k = max_num_values(len_dict)
max_v, max_k
# Key with largest number of rhyming words
rhy[max_k]

# short rhyming words
ty = []
for x in rhy:
     if len(rhy[x]) == 2:
         ty.append(rhy[x])
len(ty)
###########################################
# Build a word pronounciation dictionary
# Get homophones

# Get the reference  pronounciation dict
pro = pronounce_dict(words)

# Flip keys and values
pro_dict = keys_same_values(pro)
homop = homophones(pro_dict)
homop
###################################
# frequency distribution for each collocation
# rather than for individual words.
finder_two = nltk.collocations.BigramCollocationFinder.from_words(words)

finder_three = nltk.collocations.TrigramCollocationFinder.from_words(words)

finder_four = nltk.collocations.QuadgramCollocationFinder.from_words(words)

# Using ngram_fd, find the most common collocations in text:

finder_two.ngram_fd.most_common(10)
finder_three.ngram_fd.most_common(10)
finder_four.ngram_fd.most_common(10)

#Plot the four, three and two ngram
# Plot bigram
plt.figure(dpi=100, figsize=(7, 5))
plt.title('Fig6 Two.ngrams')
# plt.xlabel('two.ngram')
# plt.ylabel('frequency')
plt.tick_params(axis='both', which='major', labelsize=10)
rcParams.update({'figure.autolayout': True})
plt.tight_layout()
finder_two.ngram_fd.plot(10)

# Plot tri-gram
plt.figure(dpi=100, figsize=(7, 5))
plt.title('Fig7 Three.ngrams')
# plt.xlabel('three.ngram')
# plt.ylabel('frequency')
plt.tick_params(axis='both', which='major', labelsize=10)
rcParams.update({'figure.autolayout': True})
plt.tight_layout()
finder_three.ngram_fd.plot(10)

# Plot quadri-gram
plt.figure(dpi=100, figsize=(7, 5))
plt.title('Fig8 Alice four.ngram')
# plt.xlabel('three.ngram')
# plt.ylabel('frequency')
plt.tick_params(axis='both', which='major', labelsize=10)
rcParams.update({'figure.autolayout': True})
plt.tight_layout()
finder_four.ngram_fd.plot(10)
##########################################################
# POS tagging
##########################################
# Get nouns into a list from a dictionary
# nltk.help.upenn_tagset()
# Use Words which contains uppercase and lowecase words

# POS tagging alice_words list
# nltk.help.upenn_tagset()
alice_pos = nltk.pos_tag(Words)
alice_pos = list(map(list, alice_pos))

# Get nouns
n = get_nouns(alice_pos)
len(n)

# graph noun:
freqn = FreqDist(n)
freqn.plot(20, cumulative=False)

# Ten most common nouns
nq = freqn.most_common(10)
# Get a list of main characters
td = []
for a in nq:
    td.append((a[0]))
td
main_characters = ['Alice','Queen', 'King', 'Turtle', 'Gryphon', 'Hatter']


# Get only adjectives
adj = get_adj(alice_pos)
len(adj)

# Graph adjectives:
freqadj = FreqDist(adj)
freqadj.plot(20, cumulative=False)

# Ten most common adjectives
freqadj.most_common(10)

# Get only verbs
verb = get_verb(alice_pos)
len(verb)

# Graph verbs:
freqverb = FreqDist(verb)
freqverb.plot(20, cumulative=False)

# Ten most common verbs
freqverb.most_common(10)

# Find words ending in "ed" with regex
ed = [w for w in set(words) if re.search('ed$', w)]
print("There are", len(ed), "words that are in the past tense containing 'ed'")
########################################################
main_characters = ['Alice','Queen', 'King', 'Turtle', 'Gryphon', 'Hatter']
# Number of times Queen has been mentioned in each chapter
queen_count = []
for c in chapters[0:]:
    queen_count.append(c.count("Queen"))
# combine the chapter titles (keys) and “Alice” counts
# (values) and convert them to a dictionary.
mydict = dict(zip(range(1, len(chapters)+1), queen_count))
print(mydict)

plt.bar(range(1, len(chapters)+1), queen_count, edgecolor='black')
plt.title('Fig12 Distribution of "Queen" across chapters')
plt.xlabel('Chapters')
plt.ylabel('Queen count')
plt.show()

# Number of times King has been mentioned in each chapter
king_count = []
for c in chapters[0:]:
    king_count.append(c.count("King"))
# combine the chapter titles (keys) and “Alice” counts
# (values) and convert them to a dictionary.
mydict = dict(zip(range(1, len(chapters)+1), king_count))
print(mydict)

plt.bar(range(1, len(chapters)+1), king_count, edgecolor='black')
plt.title('Fig13 Distribution of "King" across chapters')
plt.xlabel('Chapters')
plt.ylabel('King count')
plt.show()

# Number of times Turtle has been mentioned in each chapter
turtle_count = []
for c in chapters[0:]:
    turtle_count.append(c.count("Turtle"))
# combine the chapter titles (keys) and “Alice” counts
# (values) and convert them to a dictionary.
mydict = dict(zip(range(1, len(chapters)+1), turtle_count))
print(mydict)

plt.bar(range(1, len(chapters)+1), turtle_count, edgecolor='black')
plt.title('Fig14 Distribution of "Turlte" across chapters')
plt.xlabel('Chapters')
plt.ylabel('Turtle count')
plt.show()

# Number of times Gryphon has been mentioned in each chapter
gryphon_count = []
for c in chapters[0:]:
    gryphon_count.append(c.count("Gryphon"))
# combine the chapter titles (keys) and “Alice” counts
# (values) and convert them to a dictionary.
mydict = dict(zip(range(1, len(chapters)+1), gryphon_count))
print(mydict)

plt.bar(range(1, len(chapters)+1), gryphon_count, edgecolor='black')
plt.title('Fig15 Distribution of "Gryphon" across chapters')
plt.xlabel('Chapters')
plt.ylabel('Gryphon count')
plt.show()

# Number of times Hatter has been mentioned in each chapter
hatter_count = []
for c in chapters[0:]:
    hatter_count.append(c.count("Hatter"))
# combine the chapter titles (keys) and “Alice” counts
# (values) and convert them to a dictionary.
mydict = dict(zip(range(1, len(chapters)+1), hatter_count))
print(mydict)

plt.bar(range(1, len(chapters)+1), hatter_count, edgecolor='black')
plt.title('Fig16 Distribution of "Hatter" across chapters')
plt.xlabel('Chapters')
plt.ylabel('Hatter count')
plt.show()

# Plot tri-gram
plt.figure(dpi=100, figsize=(7, 5))
plt.title('Fig7 Three.ngrams')
# plt.xlabel('three.ngram')
# plt.ylabel('frequency')
plt.tick_params(axis='both', which='major', labelsize=10)
rcParams.update({'figure.autolayout': True})
plt.tight_layout()
finder_three.ngram_fd.plot(10)

# Plot quadri-gram
plt.figure(dpi=100, figsize=(7, 5))
plt.title('Fig8 Alice four.ngram')
# plt.xlabel('three.ngram')
# plt.ylabel('frequency')
plt.tick_params(axis='both', which='major', labelsize=10)
rcParams.update({'figure.autolayout': True})
plt.tight_layout()
finder_four.ngram_fd.plot(10)

###############################################################
# frequency distribution of a given word
tex = nltk.Text(words)
fd = tex.vocab()
fd.tabulate(5)
# Obtain list with information about location
# of each occurrence, with text.concordance_list():
concordance_list = tex.concordance_list("alice", lines=5) 
for entry in concordance_list:
    print(entry.line)

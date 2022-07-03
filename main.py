 #!/usr/bin/python

from utils_project import *

# Read the Alice in Wonderland file from the Gutenburg repository
url = "https://www.gutenberg.org/files/11/11-0.txt"
raw = read_gutenberg(url)

# Split raw text into separate lists of chapters, titles
ct = chapter_titles(raw)

# Cleaning
text, wordlist, words_dirty, num_of_words, num_of_unique_words, fdist_Words = get_cleaned_Alice_text(ct["chapters"])

cleaned_ct = get_cleaned_Alice_text(ct["chapters"])

#Plot of frequency distribution of the most common specifed words (default = 10)
num_of_most_common_words = 10
freq_dist = freq_dist_of_n_most_common_words(num_of_most_common_words, cleaned_ct["fdist_words_dirty"])

# Plot and the number of times Alice has been mentioned in each chapter
freq_dist_alice = num_Alice_mentions(ct["chapters"])

# Word Analysis

# Count number of words  by their length.
words_sorted = count_by_wordlength(cleaned_ct["words_dirty"])

# Lexical diversity in words (only lower case alphabets)
lexical_diversity_words = lexical_diversity(words_sorted["words"])

# Lexical diversity in Words (with upper and lower case alphabets)
lexical_diversity_words = lexical_diversity(cleaned_ct["words_dirty"])

# Length of longest word
longestWord = longest_word(words_sorted["words"])
# Length of shortest word
shortestWord = shortest_word(words_sorted["words"])

# Get anagrams
anagrams = anagram(words_sorted["words"])
word_2_anagrams = anagram_count(anagrams, 2)
word_3_anagrams= anagram_count(anagrams, 3)

# Repeated 3 letter words
rep_3_words = repeated_three(words_sorted["words"])

# Find palindromes
palindromes = palindromes(words_sorted["words"])

# Find words begining with vowels
begining_with_vowels = begin_v(words_sorted["words"])

# Find words ending with vowels
ending_with_vowels = end_v(words_sorted["words"])

# dictionary with key = cv-pattern and number of words (keys)
d, key_length = dict_cv(words_sorted["words"]) 
        
# Pick up patterns with only vowels
vowel_only_patterns = patterns_only_vowels(d)

# Find words with v followed by c pattern
V_followedby_c = patterns_v_c(d)
                          
# Find words with c followed by v pattern
c_v = patterns_c_v(d)

# Sort values of dictionary d in to an array and take logarithm
logCounts = sort_values_dict_into_array(d)

# ordered dictionary with keys of the len(d[k])
# Number of  patterns are represented by one word
dict1, l = dict_with_keys_length_values(d)

# Save the sorted dictionary
Save_dict(dict1, "alice_cv.txt")

# Univocalic words containing a single type of vowel
a= univocalic_words_single_vowel(words_sorted["words"])

# Hepax Legomena :Checking for words that occur only once in the entire text
hepax_legomena_counts, hepax_legomenas = hepax_legomena_entire_text(words_sorted["words"])
# Checking for Hepax Legomena chapter by chapter
chap_hepax, per_hepax = hepax_legomena_by_chapter(cleaned_ct["wordlist"],ct["chapters"], plot=True)

# dict_rhy = a dict of longest and shortest rhyming words and their lengths
# rhy = dict of rhying words
dict_rhy, rhy = get_longest_and_shortest_rhyming_words(words_sorted["words"])

# Tuple of number of rhyming words, word with largest number of rhyming words
(max_v, max_k), rhy[max_k] = dict_number_of_rhyming_words(rhy)

# dict of homophones
homop = homophone_dict(words_sorted["words"])

# plot frequency distribution for collocation rather than for individual words.
find_collocations(words_sorted["words"])

# Get dict with lists of nouns, adjectives and verbs
dict_pos_tags = pos_tagging(cleaned_ct["words_dirty"])

# # Plot the frequency distribution of n  most common nouns, adjectives and verbs
plot_most_common_nouns_adjs_verbs(dict_pos_tags, number=10)

# List of main characters in the story
main_characters = get_most_common_main_characters(dict_pos_tags["nouns"])

# get dict of counts of main characters
main_character_counts = dict_count_of_main_characters(ct["chapters"], main_characters)

# Plot the number of times main characters across chapters
plot_main_characters(main_character_counts,ct["chapters"] )

    

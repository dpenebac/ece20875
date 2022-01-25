import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import numpy as np
from helper import remove_punc
import math

#Clean and stem the contents of a document
#Takes in a file name to read in and clean
#Return a list of words, without stopwords and punctuation, and with all words stemmed
# NOTE: Do not append any directory names to doc -- assume we will give you
# a string representing a file name that will open correctly
def read_and_clean_doc(doc) :
    #1. Open document, read text into *single* string
    f = open(doc, "r")
    data = f.read().strip(' ')

    #2. Tokenize string using nltk.tokenize.word_tokenize
    data = nltk.tokenize.word_tokenize(data)

    #3. Filter out punctuation from list of words (use remove_punc)
    i = 0
    for d in data:
        if d in string.punctuation:
            data.pop(i)
        i += 1

    #4. Make the words lower case
    data = [d.lower() for d in data]

    #5. Filter out stopwords
    filteredData = []
    stop_words = set(stopwords.words('english'))
    for d in data:
        if d not in stop_words:
            filteredData.append(d)

    #6. Stem words
    ps = PorterStemmer()
    stem = []
    for d in filteredData:
        stem.append(ps.stem(d))
    
    words = stem
    f.close()
    return words
    
#Builds a doc-word matrix for a set of documents
#Takes in a *list of filenames*
#
#Returns 1) a doc-word matrix for the cleaned documents
#This should be a 2-dimensional numpy array, with one row per document and one 
#column per word (there should be as many columns as unique words that appear
#across *all* documents. Also, Before constructing the doc-word matrix, 
#you should sort the wordlist output and construct the doc-word matrix based on the sorted list
#
#Also returns 2) a list of words that should correspond to the columns in
#docword
def build_doc_word_matrix(doclist) :
    #1. Create word lists for each cleaned doc (use read_and_clean_doc)
    clean = []

    for doc in doclist:
        clean.append(read_and_clean_doc(doc))
    clean.sort()

    wordlist = []
    for c in clean:
        wordlist = list(set(wordlist + c))
    wordlist.sort()

    #2. Use these word lists to build the doc word matrix
    docword = np.zeros((len(doclist), len(wordlist)))

    for x in range(len(docword)):
        for y in range(len(docword[x])):
            word = wordlist[y]
            count = 0
            for j in clean[x]:
                if j == word:
                    count += 1
            docword[x][y] = count

    return docword, wordlist #docword, wordlist
    
#Builds a term-frequency matrix
#Takes in a doc word matrix (as built in build_doc_word_matrix)
#Returns a term-frequency matrix, which should be a 2-dimensional numpy array
#with the same shape as docword
def build_tf_matrix(docword) :
    #fill in
    tf = np.zeros((len(docword), len(docword[0])))

    norm = []
    for row in docword:
        norm.append(sum(row))
    
    for x in range(len(tf)):
        for y in range(len(tf[x])):
            tf[x][y] = docword[x][y] / norm[x]

    return tf
    
#Builds an inverse document frequency matrix
#Takes in a doc word matrix (as built in build_doc_word_matrix)
#Returns an inverse document frequency matrix (should be a 1xW numpy array where
#W is the number of words in the doc word matrix)
#Don't forget the log factor!
def build_idf_matrix(docword) :
    #fill in
    N,X = docword.shape

    nt = [0] * len(docword[0])
    for x in range(len(docword)):
        for y in range(len(docword[x])):
            if docword[x][y] > 0:
                nt[y] += 1

    idf = []
    for i in range(len(nt)):
        idf.append(np.log10((N/ nt[i])))

    idf = np.array(idf)
    idf = idf.reshape(1, -1)
    return idf
    
#Builds a tf-idf matrix given a doc word matrix
def build_tfidf_matrix(docword) :
    #fill in
    idf = build_idf_matrix(docword)
    tf = build_tf_matrix(docword)

    tfidf = np.multiply(idf, tf)

    return tfidf
    
#Find the three most distinctive words, according to TFIDF, in each document
#Input: a docword matrix, a wordlist (corresponding to columns) and a doclist 
# (corresponding to rows)
#Output: a dictionary, mapping each document name from doclist to an (ordered
# list of the three most common words in each document
def find_distinctive_words(docword, wordlist, doclist) :
    distinctive_words = {}

    for i in range(len(doclist)):
        distinctive_words[doclist[i]] = []
    #fill in
    #you might find numpy.argsort helpful for solving this problem:
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    tfidf = build_tfidf_matrix(docword)

    for j in range(3):
        for i in range(len(doclist)):
            maxIndex = list(tfidf[i]).index(max(tfidf[i])) #index of max in tfidf
            word = wordlist[maxIndex] #word corresponding to max in tfidf
            #remove "max" from list by setting it to -1
            tfidf[i][maxIndex] = -1
            distinctive_words[doclist[i]].append(word)
            
    return distinctive_words

if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, splitext
    
    ### Test Cases ###
    directory='lecs'
    path1 = join(directory, '1_vidText.txt')
    path2 = join(directory, '2_vidText.txt')
    
    # Uncomment and recomment ths part where you see fit for testing purposes
    print("*** Testing read_and_clean_doc ***")
    print(read_and_clean_doc(path1)[0:5])


    print("*** Testing build_doc_word_matrix ***") 
    doclist =[path1, path2]
    docword, wordlist = build_doc_word_matrix(doclist)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])


    print("*** Testing build_tf_matrix ***") 
    tf = build_tf_matrix(docword)
    print(tf[0][0:10])
    print(tf[1][0:10])
    print(tf.sum(axis =1))

    
    print("*** Testing build_idf_matrix ***") 
    idf = build_idf_matrix(docword)
    print(idf[0][0:10])

    
    print("*** Testing build_tfidf_matrix ***") 
    tfidf = build_tfidf_matrix(docword)
    print(tfidf.shape)
    print(tfidf[0][0:10])
    print(tfidf[1][0:10])


    print("*** Testing find_distinctive_words ***")
    print(find_distinctive_words(docword, wordlist, doclist))
    

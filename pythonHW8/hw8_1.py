#Arguments:
#  filename: name of file to read in
#Returns: a list of strings
# each string is one line in the file, 
# and all of the characters should be lowercase, have no newlines, and have both a prefix and suffix of '__' (2 underscores)
# Example: "Hello World" should be turned into "__hello world__"
#hints: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
#       https://docs.python.org/3/library/stdtypes.html#str.splitlines
def get_formatted_text(filename) :
    #fill in
    lines = []
    f = open(filename, "r")
    for line in f:
        lines.append(line)

    i = 0
    for l in lines:
        pre = '__'
        post = '__'
        l = l.lower()
        l = l.rstrip()
        l = pre + l + post
        lines[i] = l
        i += 1

    f.close()
    return lines

#Arguments:
#  line: a string of text
#Returns: a list of 3-character n-grams
def get_ngrams(line) :
    #fill in
    n = 3
    ngrams = []
    ngrams = [line[i:i+n] for i in range(len(line) - 1)]

    return ngrams

#Arguments:
#  filename: the filename to create an n-gram dictionary for
#Returns: a dictionary
#  where ngrams are the keys and the count of that ngram is the value.
#Notes: Remember that get_formatted_text() gives you a list of lines, and you want the ngrams from
#       all the lines put together.
#       You should use get_formatted_text() and get_ngrams() in this function.
#Hint: dict.fromkeys(l, 0) will initialize a dictionary with the keys in list l and an
#      initial value of 0
def get_dict(filename):
    #fill in
    lines = get_formatted_text(filename)
    ngram_dict = {}
    for line in lines:
        ngram = get_ngrams(line)
        for i in range(len(ngram)):
            count = ngram_dict.get(ngram[i])
            if type(count) == int:
                ngram_dict[ngram[i]] = count + 1
            else:
                ngram_dict[ngram[i]] = 1
    
    return ngram_dict

#Arguments:
#   filename: the filename to generate a list of top N (most frequent n-gram, count) tuples for
#   N: the number of most frequent n-gram tuples to have in the output list.
#Returns: a list of N tuples 
#   which represent the (n-gram, count) pairs that are most common in the file.
#   To clarify, the first tuple in the list represents the most common n-gram, the second tuple the second most common, etc...
#You may find the following StackOverflow post helpful for sorting a dictionary by its values: 
#Also consider the dict method popitem()
#https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
def top_N_common(filename,N):
    common_N = []
    ngram_dict = get_dict(filename)
    q = sorted(ngram_dict.items(), key=lambda x: x[1], reverse=True)
    
    for i in range(N):
        common_N.append(q[i])
    
    return common_N

########################################## Checkpoint, can test code above before proceeding #############################################

#Arguments:
#   filenames_list: a list of filepath strings for the different language text files to process
#Returns: a list of dictionaries 
#   where each dictionary corresponds to one of the filepath strings.
#   Each dictionary in the list
#   should have keys corresponding to the n-grams, and values corresponding to the count of the n-gram
#Hint: Use functions defined in previous step.
def get_all_dicts(filenames_list):
    lang_dicts = []
    for file in filenames_list:
        ngram_dict = get_dict(file)
        lang_dicts.append(ngram_dict)
    
    return lang_dicts

#Arguments:
#   list_of_dicts: A list of dictionaries where the keys are n-grams and the values are the count of the n-gram
#Returns: an alphabetically sorted list containing all of the n-grams across all of the dictionaries in list_of_dicts (note, do not have duplicates n-grams)
#Notes: It is recommended to use the "set" data type when doing this (look up "set union", or "set update" for python)
#   Also, for alphabetically sorted, we mean that if you have a list of the n-grams altogether across all the languages, and you call sorted() on it, that is the output we want
def dict_union(list_of_dicts):
    union_ngrams = []
    combinedDict = {}

    for d in list_of_dicts:
        combinedDict.update(d)

    union_ngrams=sorted(combinedDict.keys())
    
    return union_ngrams

#Arguments:
#   lang_files: list of filepaths of the languages to compare test_file to.
#Returns a sorted list of all the n-grams across the languages
# Note: Use previous two functions.
def get_all_ngrams(lang_files):
    all_ngrams = []
    all_dicts = get_all_dicts(lang_files)
    all_ngrams = dict_union(all_dicts)

    return all_ngrams

########################################## Checkpoint, can test code above before proceeding #############################################

#Arguments:
#   test_file: mystery file's filepath to determine language of
#   lang_files: list of filepaths of the languages to compare test_file to.
#   N: the number of top n-grams for comparison
#Returns the filepath of the language that has the highest number of top 10 matches that are similar to mystery file.
#Note/Hint: depending how you implemented top_N_common() earlier, you should only need to call it once per language, and doing so avoids a possible error
def compare_lang(test_file,lang_files,N):
    lang_match = ''

    myst_top_ngrams = set([tup[0] for tup in top_N_common(test_file,N)])

    compare = []
    
    for l in lang_files:
        compare.append(set([tup[0] for tup in top_N_common(l,N)]))
    
    j = 0
    langIndex = [0] * len(lang_files)
    for gram in myst_top_ngrams:
        j = 0
        for c in compare:
            for i in c:
                if (i == gram):
                    langIndex[j] += 1
            j = j + 1

    lang_match = lang_files[max(range(len(langIndex)), key = langIndex.__getitem__)]

    return lang_match

if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, splitext
    
    #Test top_N_common()
    path = join('ngrams','english.txt')
    print(top_N_common(path,10))
    
    #Compile ngrams across all 6 languages and determine a mystery language
    path='ngrams'
    file_list = [f for f in listdir(path) if isfile(join(path, f))]
    path_list = [join(path, f) for f in file_list if 'mystery' not in f]#conditional excludes mystery.txt
    print(get_all_ngrams(path_list))#list of all n-grams spanning all languages
    
    test_file = join(path,'mystery.txt')
    print(compare_lang(test_file, path_list, 20))#determine language of mystery file
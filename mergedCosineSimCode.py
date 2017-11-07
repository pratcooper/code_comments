import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import re
import csv
import gensim
import numpy as np

# nltk.download('wordnet')
# nltk.download('punkt') # if necessary...
document1 = "This code prints hello world to the console"
document2 = "int helloWorld(){ (\" hi world \"); "
java_keyword = ['abstract','assert','boolean','break','byte','case','catch','char','class','const','continue','default','double','else','enum','extends','final','finally','float','for','if','implements','import','instanceof','int','interface','long','new','package','private','protected','public','return','short','static','super','switch','synchronized','this','throw','throws','transient','try','void','volatile','while']


'''
document3 = "This codes increase number with 1"
document4 = "public static mul (int a) { a = a + 1 }"
document5 = "This code is written at USC"
document6 = "public static void main(){ System.out.println(\"hello\")"
'''


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower().split()


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


'''remove punctuation, lowercase, stem'''

def normalize(text):
    #list_of_tok = stem_tokens(nltk.word_tokenize(text.translate(remove_punctuation_map)))
    list_of_tok = nltk.word_tokenize(text.translate(remove_punctuation_map))

    result_list_tokens = []
    #print('list of tok :', list_of_tok)
    for item in list_of_tok:
        if item not in java_keyword:
            res = convert(item)
            for i in res:
                result_list_tokens.append(i)

    #print("Final result passed to TfidfVectorizer:",result_list_tokens)
    return result_list_tokens

def raw_normalize(text):
    return stem_tokens(nltk.word_tokenize(text.translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english',lowercase = False)
raw_vectorizer = TfidfVectorizer()

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    # print(tfidf)
    return ((tfidf * tfidf.T).A)[0, 1]

def raw_cosine_sim(text1, text2):
    tfidf = raw_vectorizer.fit_transform([text1, text2])
    # print(tfidf)
    return ((tfidf * tfidf.T).A)[0, 1]

from sklearn.metrics.pairwise import cosine_similarity


# hello = wn.synset('cat.n.01')
# print (hello.lemma_names)
# print(hello[0].hypernyms())

def getSynonyms(word):
    synonymSet = set()
    for i, j in enumerate(wn.synsets(word)):
        # print ("Meaning",i, "NLTK ID:", j.name())
        # print ("Definition:",j.definition())
        # print ("Synonyms:", ", ".join(j.lemma_names()))
        for k in j.lemma_names():
            # print(k.type)
            synonymSet.add(k)

    # print (synonymSet)
    return synonymSet


def replaceSynonyms(text1, text2):
    text1 = text1.split(" ")
    text2 = text2.split(" ")
    for i in text1:
        # print (i + " is i")
        synList = getSynonyms(i)
        for j in synList:
            # print (j + " " + "is j")
            for k in range(len(text2)):
                if j == text2[k] and j != i:
                    # print("Changing here" + j + " " + i)
                    text2[k] = i

    return text1, text2

def getPair(line):

    counter = 0
    codeList = []
    commentList = []

    while(counter<len(line)):
        methodInfo = line[counter+0]
        pathToMain = line[counter+1]
        #print(line[1])
        commentLength = int(line[counter+2])

        comment = []
        for i in range(commentLength):
            comment.append(line[counter + i + 3])
        commentString = ' '.join(comment)
        commentList.append(commentString)
        codeLength = int(line[counter+ 3 + commentLength])

        code = []
        for i in range(codeLength):
            code.append(line[counter + i+4+commentLength])
        codeString = ' '.join(code)
        codeList.append(codeString)
        end = counter + commentLength+codeLength+5
        counter = end
        #print(counter)
    return codeList,commentList


'''
print("Original text")
print(document1)
print(document2)
print(cosine_sim(document1, document2))
text1, text2 = replaceSynonyms(document1, document2)
print('relaced text :' , text2)
text1 = ' '.join(text1)
text2 = ' '.join(text2)
print("Preprocessed text")
print(text1)
print(text2)

print(cosine_sim(text1, text2))
'''

# print(cosine_sim(document1, document4))
# print(cosine_sim(document1, document5))
# print(cosine_sim('a little bird', 'a little bird chirps'))
# print cosine_sim('a little bird', 'a big dog barks')


def operandSynonyms(document1, document2):
    # hashmap = { "+" : "sum" , "-" : "divides", "*" : "multiplication", "/" : "multiplication" }
    if ("+" in document2):
        # print("Here")
        document2 = document2.replace("+", "addition")
    if ("-" in document2):
        document2 = document2.replace("-", "subtract")
    if ("*" in document2):
        document2 = document2.replace("*", "multiplie")
    if ("/" in document2):
        document2 = document2.replace("/", "divide")

    # print (document2)
    return document1, document2

'''
print("Original text")
print(document3)
print(document4)
print(cosine_sim(document3, document4))
text3, text4 = operandSynonyms(document3, document4)

print(text3)
print(text4)
text3, text4 = replaceSynonyms(text3, text4)
text3 = ' '.join(text3)
text4 = ' '.join(text4)
print("Preprocessed text")
print(text3)
print(text4)
print(cosine_sim(text3, text4))
'''
#print(cosine_sim(document5, document6))

def runCosineSim(document1,document2):
    #print("Original text")
    #print(document1)
    #print(document2)
    cosine_before = cosine_sim(document1, document2)
    #print("cosine_before :" , cosine_before )
    text1, text2 = replaceSynonyms(document1, document2)
    #print('replaced text :', text2)
    text1 = ' '.join(text1)
    text2 = ' '.join(text2)
    #print("Preprocessed text")
    #print(text1)
    #print(text2)
    cosine_after  = cosine_sim(text1, text2)
    #print("cosine_after :", cosine_after)
    if cosine_before != cosine_after:
        print("Values changed")

    return cosine_after

def runRawCosineSim(document1,document2):
    raw_res_cosine = raw_cosine_sim(document1, document2)
    if raw_res_cosine == 0:
        print(document1)
        print(document2)

    return raw_res_cosine

def runCosineSimWord2Vec(document1,document2,model1,model2):

    result_vec_code = np.zeros(shape=(100,))
    for word in document1:
        try:
            word_vec = model1[word]
        except KeyError:
            word_vec = 0
        result_vec_code = np.add(result_vec_code,word_vec)

    result_vec_code = np.divide(result_vec_code,len(document1))

    result_vec_comment = np.zeros(shape=(100,))
    for word in document2:
        try:
            word_vec = model2[word]
        except KeyError:
            word_vec = 0
        result_vec_comment = np.add(result_vec_comment, word_vec)

        result_vec_comment = np.divide(result_vec_comment, len(document2))

    numerator = np.dot(result_vec_code,result_vec_comment)
    dinominator = np.multiply(np.linalg.norm(result_vec_code),np.linalg.norm(result_vec_comment))

    cosine_similar= float(numerator/dinominator)

    return cosine_similar

if __name__=="__main__":
    resCosineValList = []
    resRawCosineValList = []
    with open("/Users/prathameshnaik/PycharmProjects/DR_Code/tp") as f:
        lines = f.readlines()

        codeList, commentList = getPair(lines)
        cnt = 1

        #model word2Vec
        #model = gensim.models.Word2Vec(sentences, min_count=1)
        sentences = ""
        for i in codeList:
            sentences = sentences + i

        model1 = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
        sentences = ""

        for i in commentList:
            sentences = sentences + i

        model2 = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)


        for i,j in zip(codeList,commentList):
            print('dataset :' ,cnt)
            cnt = cnt + 1
            res_cosine = runCosineSim(i, j)
            resCosineValList.append(res_cosine)
            raw_res_cosine = runRawCosineSim(i,j)
            resRawCosineValList.append(raw_res_cosine)
            res_cosine_word2vec = runCosineSimWord2Vec(i,j,model1,model2)

            print("Raw value:",raw_res_cosine)
            print("Modified value using features:", res_cosine)
            print("Modified value using word2vec:", res_cosine_word2vec)
            print("-----------")

    with open("/Users/prathameshnaik/PycharmProjects/DR_Code/data.csv",'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i,j in zip(resCosineValList,resRawCosineValList):
            wr.writerow([float(i),float(j)])



# print (text3)
# print (text4)
# print(cosine_sim(text3, text4))
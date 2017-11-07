import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import re

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

    print("Final result passed to TfidfVectorizer:",result_list_tokens)
    return result_list_tokens


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english',lowercase = False)


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
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




# print("Preprocessed text")
# print (text3)
# print (text4)
# print(cosine_sim(text3, text4))
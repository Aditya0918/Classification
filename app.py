from flask import Flask,jsonify,request
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re
import tensorflow
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.stem import WordNetLemmatizer 
sentence_length=685

app = Flask(__name__)

with open('tokenizer.pkl','rb') as f:

	tokens=pickle.load(f)

print("Vocabulary loaded")

predictor = load_model("network.h5")

print("Model loaded")

@app.route('/',methods=['GET'])
def home():

    text=str(request.args['text'])
    #nltk.download('stopwords')
    text_list=text.split('.')
    stop_words = stopwords.words('english')
    if len(text_list)<=5:

        predict_class(text)

    elif len(text_list)>5:

        summarize_text = []
        decimal_nums=re.findall('\d{1}\d*\.\d{1}\d*',text)
        list_of_words=text.split()
        new_list_of_words=[]
        for word in list_of_words:

            if word in decimal_nums:

                temp_number=re.sub('\.','\|',word)
                new_list_of_words.append(temp_number)

            else:

                new_list_of_words.append(word)

        document=' '.join(new_list_of_words)       
        sentences =  read_article(document)

        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        print(ranked_sentence)

        if len(ranked_sentence)<=10:

            top_n=5

        else:

            top_n=10

        for i in range(top_n):
                
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        final_summary=". ".join(summarize_text)
        final_summary=final_summary.replace("\\|",'.')
        predict_class(final_summary)

    def read_article(ocr_text):

        article = ocr_text.split(".")
        sentences = []

        for sentence in article:
            print(sentence)
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop() 
            
        return sentences

    def sentence_similarity(sent1, sent2, stopwords=None):

        if stopwords is None:
            stopwords = []
        
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
        
        all_words = list(set(sent1 + sent2))
        
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
            
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1
        

        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1
        
        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(sentences, stop_words):
        
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] =sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix


def predict_class(input_txt):


    x=str(input_txt)
    x=re.sub('pic.twitter.com\/[A-Za-z0-9]{1}[A-Za-z0-9]*',' ',x)
    x=re.sub('#[A-Za-z0-9]{1}[A-Za-z0-9]*',' ',x)
    x=re.sub(r'http\S+',' ',x)
    x=re.sub('IANS[A-Za-z]{3}\/[A-Za-z]{3}',' ',x)
    x=re.sub('IANS\s[A-Za-z]{3}\/[A-Za-z]{3}\/',' ',x)
    x=re.sub('IANS[A-Za-z]{2}\/[A-Za-z]{3}',' ',x)
    x=re.sub('IANS\s[A-Za-z]{2}\/[A-Za-z]{3}',' ',x)
    x=re.sub('\(IANS Interview\)',' ',x)
    x=re.sub('IANS[A-Za-z]{2}\/[A-Za-z]{3}\/[A-Za-z]{3}\/[A-Za-z]{2}',' ',x)
    x=re.sub('IANS\s[a-zA-Z]{3}',' ',x)
    x=re.sub('IANS\/GloFansqma\/',' ',x)
    x=re.sub('IANS\s[a-zA-Z]{2}\/[a-zA-Z]{2}',' ',x)
    x=re.sub('IANS[a-zA-Z]{3}\/[a-zA-Z]{2}\/[a-zA-Z]{3}',' ',x)
    x=re.sub('\(Column: Close-in\)',' ',x)
    x=re.sub('Goalscorer','goal scorer',x)
    x=re.sub('Batswoman','bats woman',x)
    x=re.sub('Oppn','opposition',x)
    x=re.sub('Govt','government',x)
    x=re.sub('JD\(U\)','JDU',x)
    x=re.sub('CPI\s\(M\)','CPI-M',x)
    x=re.sub('Prof','Professor',x)
    x=re.sub('JD-U','JDU',x)
    x=re.sub('Cong','Congress',x)
    x=re.sub('CPI-M','CPI\(M\)',x)
    x=re.sub('UP','Uttar Pradesh',x)
    x=re.sub('I-T','IT',x)
    x=re.sub('J-K','JK',x)
    x=re.sub('Agusta Westland','AgustaWestland',x)
    x=re.sub("370",'three seventy',x)
    x=re.sub("375",'three seventy five',x)
    x=re.sub("15",'fifteen',x)
    x=re.sub('Log 9','Log9',x)
    x=re.sub('billion','bn',x)
    x=re.sub('per\scent','percent',x)
    x=re.sub('mmscmd',' mmscmd',x)
    x=re.sub('million','mn',x)
    x=re.sub('trillion','tn',x)
    x=re.sub('mAh',' mAh',x)
    x=re.sub('Nfty','Nifty',x)
    x=re.sub('p\/ltr',' p/ltr',x)
    x=re.sub('Ear 1','Ear1',x)
    x=re.sub('%',' percent',x)
    #x=re.sub('S Korea','South Korea',x)
    x=re.sub('IANS\s[A-Za-z]{2}\/[A-Za-z]{2}\/[A-Za-z]{3}',' ',x)
    x=re.sub('Nm',' Nm',x)
    x=re.sub('PS', ' PS',x)
    x=re.sub('Pulsar 125','Pulsar125',x)
    x=re.sub('start-up','start up',x)
    x=re.sub('a\/c','account',x)
    x=re.sub('q-o-q','qoq',x)
    x=re.sub('yr','year',x)
    x=re.sub('a\.m\.',' am',x)
    x=re.sub('p\.m\.',' pm',x)
    x=re.sub('Larsen & Toubro','L&T',x)
    x=re.sub('kV',' kV',x)
    x=re.sub('G-7','G7',x)
    x=re.sub('G-20','G20',x)
    x=re.sub('km/kg',' km/kg',x)
    x=re.sub('mmHg',' mmHg',x)
    x=re.sub('kPa',' kPa',x)
    x=re.sub("www\.{1}.*\.com","",x)
    x=re.sub('3037 TX','3037TX',x)
    
    for match in re.finditer('[A-Z]\.',x):
    
        #start_index=match.start()
        pattern=match.group()
        list_of_letters=pattern.split('.')[:-1]
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)
        
    for match in re.finditer('FY\s\d{4}',x):
    
        #start_index=match.start()
        pattern=match.group()
        print(pattern)
        list_of_letters=pattern.split()
        print(list_of_letters)
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)
    
    for match in re.finditer('FY\s\d{2}',x):
    
        #start_index=match.start()
        pattern=match.group()
        print(pattern)
        list_of_letters=pattern.split()
        print(list_of_letters)
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)
        
    for match in re.finditer('Windows\s\d{2}',x):
    
        #start_index=match.start()
        pattern=match.group()
        print(pattern)
        list_of_letters=pattern.split()
        print(list_of_letters)
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)
        
    for match in re.finditer('[A-Z]{1}\s[A-Z]{1}\s',x):
        
        pattern=match.group()
        list_of_letters=pattern.split()
        list_of_letters.append(' ')
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)
        
    for match in re.finditer('[A-Z]{1}\s[A-Z]{1}\s[A-Z]{1}\s',x):
        
        pattern=match.group()
        list_of_letters=pattern.split()
        list_of_letters.append(' ')
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)

    for match in re.finditer('[A-Z]{1}\s[A-Z]{1}\s',x):
        
        pattern=match.group()
        list_of_letters=pattern.split()
        list_of_letters.append(' ')
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)
        
    for match in re.finditer('[A-Z]{1}\s[A-Z]{1}\s[A-Z]{1}\s',x):
        
        pattern=match.group()
        list_of_letters=pattern.split()
        list_of_letters.append(' ')
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)
        
    for match in re.finditer('[A-Z]\.',x):
    
        #start_index=match.start()
        pattern=match.group()
        print(pattern)
        list_of_letters=pattern.split('.')[:-1]
        replacing_string=''.join(list_of_letters)
        x=x.replace(pattern,replacing_string)
        
    x=re.sub('[^a-zA-Z]',' ',str(x))
    x=x.lower()
    x=x.split()
    lemmatizer = WordNetLemmatizer()
    x=[lemmatizer.lemmatize(word,"v") for word in x if not word in set(stopwords.words('english'))]
    x=' '.join(x)
    x=[x]
    x=tokens.texts_to_sequences(x)
    x=pad_sequences(x,padding="post",maxlen=sentence_length)
    x=np.array(x)
    genre=predictor.predict(x)
    print(genre)
    genre=genre.tolist()
    genre=genre[0]
    result = genre.index(max(genre))
    print(genre)
    print(result)

    if result==0:
        
        category="Business"
        
    elif result==1:
        
        category="Entertainment"
        
    elif result==2:
        
        category="Health"
        
    elif result==3:
        
        category="Politics"
        
    elif result==4:
        
        category="Sports"

    return jsonify({'Summary': category})








    
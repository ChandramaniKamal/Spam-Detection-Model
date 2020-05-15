# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:51:03 2019

@author: dell
"""

import pandas as pd
import nltk
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn import preprocessing 
from sklearn.feature_selection import chi2, SelectKBest
import re 
import emoji 
from sklearn import model_selection
import textblob
from sklearn import naive_bayes, svm, linear_model
from nltk.tokenize import sent_tokenize
import math
import sys
import numpy
import matplotlib.pyplot as plt
sys.path.append('../')
from NiaPy.algorithms.basic import KrillHerdV1,KrillHerdV2,KrillHerdV3,KrillHerdV4,KrillHerdV11
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere, SumSquares, Zakharov, Ackley, Alpine1, BentCigar, ChungReynolds, CosineMixture
from NiaPy.benchmarks import Csendes, Discus, DixonPrice, Griewank, HappyCat, HGBat, Infinity, Levy, Pinter, Powell
from NiaPy.benchmarks import Qing
from NiaPy.algorithms.algorithm import Algorithm
from numpy import apply_along_axis
from NiaPy.util import FesException, GenException, TimeException, RefException
from math import inf
from numpy import argmin, ndarray, full, transpose, argmax, asarray
from NiaPy.algorithms.algorithm import Individual
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from numpy import exp, pi, cos, sqrt, sin

def Fun(sol):
    val = 0.0
    for i in range(6305):
        val += math.pow(math.pow(sol[i], 2) - i, 2)

    return val
	
def getBest_algo(X, X_f, xb=None, xb_f=inf):
		ib = argmin(X_f)
		if isinstance(X_f, (float, int)) and xb_f >= X_f: xb, xb_f = X, X_f
		elif isinstance(X_f, (ndarray, list)) and xb_f >= X_f[ib]: xb, xb_f = X[ib], X_f[ib]
		return (xb.x.copy() if isinstance(xb, Individual) else xb.copy()), xb_f
    
def population_algo(task, matrix):
        pop=matrix
        fpop=apply_along_axis(Fun,1,pop)
        return pop,fpop

def initPopulation_algo(task, matrix):
		pop, fpop = population_algo(task, matrix)
		return pop, fpop, {}

def run_algo(kh, task, KH, KH_f, xb, fxb, W_n, W_f, N, F):
		try:
			# task.start()
			r = runTask_KH(kh, task, KH, KH_f, xb, fxb, W_n, W_f, N, F)
			return r[0], r[1] * task.optType.value
		except (FesException, GenException, TimeException, RefException): return task.x, task.x_f * task.optType.value
		except Exception as e: print(e)
		return None, None
		
def runTask_KH(kh, task, KH, KH_f, xb, fxb, W_n, W_f, N, F):
    s=[]
    d={'KH':KH, 'KH_f':KH_f, 'xb':xb, 'fxb':fxb, 'W_n':W_n, 'W_f':W_f, 'N':N, 'F':F}
    s.append(d)
    r1=[]
    r2=[]
    count=0
    while not task.stopCond():
        dt=s.pop()
        KH1, KH_f1, xb1, fxb1, D1=runIteration_KH(kh, task, dt['KH'], dt['KH_f'], dt['xb'], dt['fxb'], dt['W_n'], dt['W_f'], dt['N'], dt['F'])
        d['KH'], d['KH_f'], d['xb'], d['fxb'], d['W_n'], d['W_f'], d['N'], d['F'] = KH1, KH_f1, xb1, fxb1, D1['W_n'], D1['W_f'], D1['N'], D1['F']
        print(count,"------->  xb=",xb1," fxb = ",fxb1)
        s.append(d)
        r1.append(count)
        r2.append(fxb)
        count=count+1
        task.nextIter()
    return d['xb'],d['fxb']

def initPopulation_KH(task, matrix , kh):
	KH, KH_f, d = initPopulation_algo(task, matrix)
	W_n, W_f = kh.initWeights(task)
	N, F = full(kh.NP, .0), full(kh.NP, .0)
	d.update({'W_n': W_n, 'W_f': W_f, 'N': N, 'F': F})
	return KH, KH_f, d

def getFoodLocation_KH(kh, KH, KH_f, task):
	x_food = task.repair(asarray([sum(KH[:, i] / KH_f) for i in range(task.D)]) / sum(1 / KH_f), rnd=kh.Rand)
	x_food_f = Fun(x_food)
	return x_food, x_food_f
    
def runIteration_KH(kh, task, KH, KH_f, xb, fxb, W_n, W_f, N, F):
    ikh_b, ikh_w = argmin(KH_f), argmax(KH_f)
    #print(ikh_b)
    #print(ikh_w)
    x_food, x_food_f = getFoodLocation_KH(kh, KH, KH_f, task)
    #print(x_food)
    #print(x_food_f)
    if x_food_f < fxb: xb, fxb = x_food, x_food_f  # noqa: F841
    #print(xb)
    #print(fxb)
    N = asarray([kh.induceNeighborsMotion(i, N[i], W_n, KH, KH_f, ikh_b, ikh_w, task) for i in range(kh.NP)])
    #print(N)
    F = asarray([kh.induceForagingMotion(i, x_food, x_food_f, F[i], W_f, KH, KH_f, ikh_b, ikh_w, task) for i in range(kh.NP)])
    #print(F)
    #print(F.shape)
    D = asarray([kh.inducePhysicalDiffusion(task) for i in range(kh.NP)])
    #print(D)
    #print(D.shape)
    KH_n = KH + (kh.deltaT(task) * (N + F + D))
    Cr = asarray([kh.Cr(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) for i in range(kh.NP)])
    #print(Cr)
    #print(Cr.shape)
    KH_n = asarray([kh.crossover(KH_n[i], KH[i], Cr[i]) for i in range(kh.NP)])
    Mu = asarray([kh.Mu(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) for i in range(kh.NP)])
    #print(Mu)
    #print(Mu.shape)
    KH_n = asarray([kh.mutate(KH_n[i], KH[ikh_b], Mu[i]) for i in range(kh.NP)])
    #print(KH_n)
    #print(KH_n.shape)
    KH = apply_along_axis(task.repair, 1, KH_n, rnd=kh.Rand)
    KH_f = apply_along_axis(Fun, 1, KH)
    xb, fxb = getBest_algo(KH, KH_f, xb, fxb)
    return KH, KH_f, xb, fxb, {'W_n': W_n, 'W_f': W_f, 'N': N, 'F': F}
    
def Krill_Herd_Optimization_4(matrix):
        task = StoppingTask(D=6305, nGEN=1, benchmark=Qing())
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
        #matrix=np.transpose(matrix)
        kh=KrillHerdV4(NP=3001)
        
        KH, KH_f, d = initPopulation_KH(task, matrix, kh)
        #print(KH)
        #print(KH_f)
        #print(KH_f.shape)
        #print(d)
        xb, fxb=getBest_algo(KH,KH_f)
        print(xb)
        #print(xb.shape)
        print(fxb)
        xb, fxb=run_algo(kh, task, KH, KH_f, xb, fxb, d['W_n'], d['W_f'], d['N'], d['F'])
        return xb,fxb
    

def rapid_miner(matrix):
        for i in range(len(matrix)):
            val=0
            count=0
            for j in range(len(matrix[i])):
                if matrix[i][j]!=0.0 and numpy.isnan(matrix[i][j])==False:
                    count+=1
            
            for j in range(len(matrix[i])):
                if matrix[i][j]!=0.0 and numpy.isnan(matrix[i][j])==False:
                    val+=(math.pow(matrix[i][j],2))
                
            val=math.sqrt(val)
            
            for j in range(len(matrix[i])):
                if matrix[i][j]!=0.0 and numpy.isnan(matrix[i][j])==False:
                    matrix[i][j]=matrix[i][j]/val
                
        return matrix

def signals(arr,word_weight,vocab,l):
        max=-inf
        min=inf
        avg=0
        count=0
        for i in l:
            if arr[i]!=0.0 and numpy.isnan(arr[i])==False:
                for key, value in vocab.items():
                    if value==i:
                        arr[i]=arr[i]*word_weight[key]
                if arr[i]>max:
                    max=arr[i]
                if arr[i]<min:
                    min=arr[i]
                avg+=arr[i]
                count+=1
        if count!=0:
            avg = avg/count
        return max, avg, min

def output_signals(PAMP, Danger, Safe):
        CSM = 0.1*PAMP + 1.0*Danger + 0.0*Safe
        smDC= 0.0*Safe+0.0
        mDC = 1.0*PAMP + 1.0*Danger + 1.0*Safe
        return CSM, smDC, mDC

def MCAV_count(arr,mat_atgn_con):
        count=0
        count1=0
        for i in range(len(arr)):
            
            if arr[i]!=0 and numpy.isnan(arr[i])==False and arr[i]>mat_atgn_con:
                count+=1
            if arr[i]!=0 and numpy.isnan(arr[i])==False:
                count1+=1
                
        return count,count1
    
def DCA_Algorithm(data):
    
        df=data
		train_x = df.loc[:3000,'_message_'].values
		train_y = df.loc[:3000,'tag'].values

		valid_x=df.loc[3001:,'_message_'].values
		valid_y=df.loc[3001:,'tag'].values

		encoder = preprocessing.LabelEncoder()
		train_y = encoder.fit_transform(train_y)
		valid_y = encoder.fit_transform(valid_y)

		#Using RapidMiner to enhance Tf matrix

		#Using tfidf vectorizer
		"""
		tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
		tfidf_vect.fit(train_x)
		#print(tfidf_vect.vocabulary_)
		matrix_train= tfidf_vect.transform(train_x).toarray()

		"""
		#Using count vectorizer
		count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
		count_vect.fit(train_x)
		#print(count_vect.vocabulary_)
		matrix_train=count_vect.transform(train_x).toarray()

		matrix_train=np.array(matrix_train, dtype=float)
		print(matrix_train.shape)

		xb,fxb=Krill_Herd_Optimization_4(matrix_train)
		#print(xb)
		#print(fxb)
           
        #print(xb.shape)
		#print(fxb)
		#print(xb.min())
		#print(xb.max())
		#print(xb.max()-xb.min())

		dt=defaultdict(list)
		d={}

		count=0
		for i in range(len(xb)):
			ind=int(((xb[i]-xb.min())/(xb.max()-xb.min()))*1000)
			for key,value in count_vect.vocabulary_.items():
				if value == i:
					dt[ind].append(key)
			d[i]=xb[i]
			
		d={k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
		#print(d)

		x=[1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6305]
		y=[5305,4805,4305,3805,3305,2805,2305,1805,1305,805,305,0]
		z=[]
		Ham_msg=[]
		Spam_msg=[]
		for j in y:
			l=[]
			i=0
			for k,v in d.items():
				if i<=j:
					i=i+1
					continue
				else:
					l.append(k)
				i=i+1
				#for key,value in count_vect.vocabulary_.items():
					#if value==k:
						#print("i = ",i," key = ",key," val = ",value)
			word_weight={}
			for key,value in count_vect.vocabulary_.items():
				if value in l:
					word_weight[key]=0

			Hc=0.1
			Sc=0.5
			for i in range(len(matrix_train)):
						for j in l:
							if matrix_train[i][j]!=0.0 and numpy.isnan(matrix_train[i][j])==False:
								for key, value in count_vect.vocabulary_.items():
										if value == j:
											if train_y[i]==0:
												word_weight[key]-=Hc
											else:
												word_weight[key]+=Sc
			#print(word_weight)    
			vector = count_vect.transform(valid_x)
			matrix=vector.toarray()
			matrix = np.array(matrix, dtype=float)
			matrix=rapid_miner(matrix)
			predictions=[]
			ham_count=0
			spam_count=0

			for i in range(len(matrix)):
				res=0
				PAMP, Danger, Safe=signals(matrix[i],word_weight,count_vect.vocabulary_,l)
				#print((3001+i) , "PAMP= ",PAMP," Danger= ",Danger," Safe= ",Safe," Result= ",valid_y[i])
				if PAMP==-inf or Danger==0.0 or Safe==inf:
					res=0
					if valid_y[i]==0:
						ham_count+=1
					else:
						spam_count+=1
				else:
					for j in l:
						if matrix[i][j]!=0.0 and numpy.isnan(matrix[i][j])==False:
							for key, value in count_vect.vocabulary_.items():
								if value == j:
									if valid_y[i]==0:
										word_weight[key]-=Hc
									else:
										word_weight[key]+=Sc

					CSM, smDC, mDC =output_signals(PAMP, Danger, Safe)
					if mDC>smDC:
						res=1
					else:
						res=0
				predictions.append(res)
			#print(ham_count, spam_count)
			#print(accuracy_score(valid_y, predictions),classification_report(valid_y, predictions),confusion_matrix(valid_y, predictions))
			z.append(recall_cal(confusion_matrix(valid_y,predictions)))
			Ham_msg.append(ham_count)
			Spam_msg.append(spam_count)
			for i in range(len(x)):
				print(x[i], "-->" , z[i]," not recognized Ham messages = ", Ham_msg[i], " Spam Messages= ", Spam_msg[i])
			plt.plot(x, z)
			plt.xlabel('Iteration')
			plt.ylabel('Result')
			plt.title('Performance Graph')
			plt.show()
            
            
    #        mat_atgn_con=0
    #        Tm=0.25
    #        mature_count,antigen_count=MCAV_count(matrix[i],mat_atgn_con)
    #        MCAV=mature_count/antigen_count
    #        if res==1:
    #            print(valid_y[i])
    #            print(valid_x[i])
    #            MCAV=MCAV_count(matrix[i],mat_atgn_con)
    #            if MCAV>Tm:
    #                print("Message is anomalous with high concentration", MCAV)
    #            else:
    #                print("Message is anomalos with low concentration", MCAV)
    
        return accuracy_score(valid_y, predictions),classification_report(valid_y, predictions),confusion_matrix(valid_y, predictions)

def find_url(string): 
        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
        if url:
            return 1
        else:
            return 0

def vocab_spam_words(message,vocab):
        tokens=word_tokenize(message)
        #Now remove stop words
        stop_words=set(stopwords.words("english"))
        filtered_tokens=[]
        for w in tokens:
            if w not in stop_words:
                filtered_tokens.append(w)
                
        #Now use stemming
        lem = WordNetLemmatizer()
        stemmed_words=[]
        for w in filtered_tokens:
            stemmed_words.append(lem.lemmatize(w))
        
        for i in stemmed_words:
            if i not in vocab.keys():
                vocab[i]=1
        return tokens,vocab

def check_pos_tag(x, flag):
        pos_family = {
            'noun' : ['NN','NNS','NNP','NNPS'],
            'pron' : ['PRP','PRP$','WP','WP$'],
            'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj' :  ['JJ','JJR','JJS'],
            'adv' : ['RB','RBR','RBS','WRB']
        }
        cnt = 0
        try:
            wiki = textblob.TextBlob(x)
            for tup in wiki.tags:
                ppo = list(tup)[1]
                if ppo in pos_family[flag]:
                    cnt += 1
        except:
            pass
        return cnt

def extract_emoji(string):
        count=0
        for c in string:
            if c in emoji.UNICODE_EMOJI:
                count=count+1
        return count

def extract_special_char(string):
        count=0
        regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]') 
        for c in string:
            if(regex.search(c) != None):
                count=count+1
        return count

def feature_engg(train_x, valid_x, train_y, valid_y):
    
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(train_x)
        
        xtrain_tfidf =  tfidf_vect.transform(train_x)
        xvalid_tfidf =  tfidf_vect.transform(valid_x)
        
        
        """
        df_vec=pd.DataFrame(tfidf_vect.transform(df._message_).todense(),columns=tfidf_vect.get_feature_names())
        df=pd.concat([df,df_vec],axis=1)
    
        df['char_count'] = df['_message_'].apply(len)
        df['word_count'] = df['_message_'].apply(lambda x: len(x.split()))
        df['word_density'] = df['char_count'] / (df['word_count']+1)
        
        df['noun_count'] = df['_message_'].apply(lambda x: check_pos_tag(x, 'noun'))
        df['verb_count'] = df['_message_'].apply(lambda x: check_pos_tag(x, 'verb'))
        df['adj_count'] = df['_message_'].apply(lambda x: check_pos_tag(x, 'adj'))
        df['adv_count'] = df['_message_'].apply(lambda x: check_pos_tag(x, 'adv'))
        df['pron_count'] = df['_message_'].apply(lambda x: check_pos_tag(x, 'pron'))
        
        df['emoji']=df['_message_'].apply(lambda x: extract_emoji(x))
        df['URL']=df['_message_'].apply(lambda x: find_url(x))
        df['sp_chars']=df['_message_'].apply(lambda x: extract_special_char(x))
        
        #print(df)
        """
        
        return xtrain_tfidf, xvalid_tfidf, train_y, valid_y
   
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
        classifier.fit(feature_vector_train, label)
        
        predictions = classifier.predict(feature_vector_valid)
        
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)
        
        return accuracy_score(valid_y, predictions),classification_report(valid_y, predictions),confusion_matrix(valid_y, predictions)

def recall_cal(con_mat):
        return con_mat[1][1]/(con_mat[1][0]+con_mat[1][1]) 

def model(data):
    
        df=data
        train_x1 = df.loc[:3000,'_message_'].values
        train_y1 = df.loc[:3000,'tag'].values
        
        valid_x1=df.loc[3001:,'_message_'].values
        valid_y1=df.loc[3001:,'tag'].values
        
        encoder = preprocessing.LabelEncoder()
        train_y1 = encoder.fit_transform(train_y1)
        valid_y1 = encoder.fit_transform(valid_y1)
        
        train_x, valid_x, train_y, valid_y = feature_engg(train_x1, valid_x1, train_y1, valid_y1)
        performance=[]
        
        #Using naive_bayes
        accuracy,classification_report,confusion_matrix = train_model(naive_bayes.MultinomialNB(), train_x, train_y, valid_x, valid_y)
        print("NB, WordLevel TF-IDF: ", accuracy)
        print(confusion_matrix)
        print(classification_report)
        performance.append(recall_cal(confusion_matrix))
        
        #Logistic regression
        accuracy,classification_report,confusion_matrix  = train_model(linear_model.LogisticRegression(solver='lbfgs', max_iter=1000), train_x, train_y, valid_x, valid_y)
        print ("LR, WordLevel TF-IDF: ", accuracy)
        print(confusion_matrix)
        print(classification_report)
        performance.append(recall_cal(confusion_matrix))
        
        #SVM
        accuracy,classification_report,confusion_matrix  = train_model(svm.SVC(gamma=0.22), train_x, train_y, valid_x, valid_y)
        print ("SVM, WordLevel TF-IDF: ", accuracy)
        print(confusion_matrix)
        print(classification_report)
        performance.append(recall_cal(confusion_matrix))
        
        #DCA_Algorithm
        accuracy,classification_report,confusion_matrix=DCA_Algorithm(data)
        print ("DCA Algorithm: ", accuracy)
        print(confusion_matrix)
        print(classification_report)
        performance.append(recall_cal(confusion_matrix))
        
        #plot the graph
        models=('NB','LR','SVM','DCA')
        y_pos = np.arange(len(models))
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, models)
        plt.ylabel('Performance')
        plt.title('Models')
        plt.show()
    
    
    
if __name__ == "__main__":
        #Read the data from file
        filename='E:/Minor and Major Project/smsspamcollection/SMSSpamCollection'
        names= ['tag','_message_']
        data=pd.read_csv(filename,sep="\t",names=names)
            
        #Try different models and check accuracy
        model(data)
        
        #DCA_Algorithm(data)
    
    
       
    
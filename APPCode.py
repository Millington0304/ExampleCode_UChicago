##This is originally a notebook, now merged into one .py file for submission purpose
##The debug or test code and comments are omitted here

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy

db_path="PATH HIDDEN HERE"
df=pd.read_csv(db_path)
df=df[:1000]

df.info()

from spacy.lang.en.stop_words import STOP_WORDS
import string

punctuations=string.punctuation
stopwords=list(STOP_WORDS)

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([word for word in mytokens])
    return mytokens

from tqdm import tqdm
tqdm.pandas()
df['processed_text']=df['content'].progress_apply(spacy_tokenizer)

from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text, maxi_features):
    vectorizer=TfidfVectorizer(max_features=maxi_features)
    ret=vectorizer.fit_transform(text)
    return ret

text=df['processed_text'].values
max_feat=8192
X=vectorize(text,max_feat)
X.shape

XA=X.toarray()
for i in range(len(XA)):
    if XA[i].max()==0:
        print(i)

from sklearn.decomposition import PCA
pca=PCA(n_components=128)
pca.fit(XA)
XA=pca.transform(XA)


import random
import time
q=0.30
wl=32
cnum=10
csize=5
rsize=5
objpsize=100

def dissimilarity(vectors):
    s=0
    for i in range(len(vectors)):
        for j in range(i+1,len(vectors)):
            
            
            
            linearDis=1-cos_similarity(vectors[i],vectors[j])
            s+=4-(linearDis-1)**2
    s/=(len(vectors)**2-len(vectors))/2
    return s

def cos_similarity(a,b):
    if(np.linalg.norm(a,2)*np.linalg.norm(b,2)==0):
        return 0
    return np.dot(a,b)/np.linalg.norm(a,2)/np.linalg.norm(b,2)


def total_similarity(a,b):
    acenter=np.average(a,axis=0)
    s=0
    for i in b:
        s+=2**cos_similarity(acenter,i)-0.5
    s/=len(b)
    return s

def objective_function(recommendation,original):
    t=total_similarity(original,recommendation)*q+(1-q)*dissimilarity(recommendation)
    if(t<0):
        print(t)
    return total_similarity(original,recommendation)*q+(1-q)*dissimilarity(recommendation)


def get_score(w,ind,a,b):
    s=0
    for wi,i in enumerate(ind):
        s+=a[i]*b[i]*w[wi]
    return s

def one_normalization(w):
    s=0
    for i in w:
        s+=abs(i)
    return [i/s for i in w]

def recommend(w,original,num):
    ocenter=np.average(original,axis=0)
    largest_indices=np.array(ocenter).argsort().tolist()[::-1]
    largest_indices=largest_indices[0:wl]
    sw=0
    so=0
    for wi,i in enumerate(largest_indices):
        sw+=abs(w[wi])
        so+=abs(ocenter[i])
    w0=[i/sw for i in w]
    ocenter0=[i/so for i in ocenter]
    ss=np.zeros(len(XA))
    
    
    for i in random.choices(range(len(XA)),k=objpsize):
        ss[i]=get_score(w0,largest_indices,XA[i],ocenter0)
        for t in original:
            if((XA[i] == t).all()):
                ss[i]=-32768
    return ss.argsort().tolist()[-num:]

def recommendi(w,original,num):
    w0=one_normalization(w)
    ss=np.zeros(len(XA))
    for i in range(len(XA)):
        slist=[]
        for j in original:
            largest_indices=np.array(j).argsort().tolist()[::-1]
            slist.append(get_score(w0,largest_indices[0:wl],XA[i],j))
        ss[i]=sum(sorted(slist,reverse=True)[:int(len(original)/3)])
    return ss.argsort().tolist()[0:num]


def obtainc(num,siz,rseed):
    a=[]
    b=[]
    random.seed(rseed)
    for i in range(num):
        
        a.append(random.choices(range(len(XA)),k=siz))
        b.append([XA[j] for j in a[i]])
    return b,a


pop_in=100
pop_out=pop_in*2
ini_low=-0.5
ini_high=1
population=np.zeros((pop_out,wl))
c=[]
ci=[]
cc=[]
best=[]
bestw=np.zeros(wl)

def initGA():
    global c,ci,cc
    c,ci=obtainc(cnum,csize,0xa676c294)
    cc=[np.average(i,axis=0) for i in c]
    
    for i in range(pop_in):
        for j in range(wl):
            population[i][j]=np.random.rand()*(ini_high-ini_low)+ini_low

def o(w):
    s=0
    for i in c:
        s+=objective_function(XA[recommend(w,i,rsize)],i)
    return s

def os(w):
    s=0
    for i in c:
        
        s+=total_similarity(i,XA[recommend(w,i,rsize)])
    return s

def od(w):
    s=0
    for i in c:
        s+=dissimilarity(XA[recommend(w,i,rsize)])
    return s

def selectParents():
    parents=np.zeros((pop_out,2))
    prob=np.zeros(pop_in)
    for i in range(pop_in):
        prob[i]=o(population[i])
    parents=np.array(random.choices(range(pop_in),prob,k=2*pop_out)).reshape((pop_out,2))
    return parents

def randmix(a,b):
    c=b
    for i in range(a.shape[0]):
        if np.random.rand()>0.5:
            c[i]=b[i]
    return c

def randmix_2(a,b,lst):
    c=[]
    i=0
    flag=True
    lst.append(len(a))
    for p in sorted(lst):
        if(flag):
            c.extend(a[i:p])
            i=p
        else:
            c.extend(b[i:p])
            i=p
        flag=not flag
    return c




def crossover(parents):
    global population
    tmpPopu=np.zeros((pop_out,wl))
    for i,par in enumerate(parents):
        tmpPopu[i]=randmix(population[par[0]],population[par[1]])
    population=tmpPopu

def crossover_2(parents):
    global population
    tmpPopu=np.zeros((pop_out,wl))
    for i,par in enumerate(parents):
        tmpPopu[i]=randmix_2(population[par[0]],population[par[1]],random.choices(range(0,len(population[par[0]])),None,k=random.randint(1,wl//4)))
    population=tmpPopu

def mutate():
    global population
    chance=40/1000
    d=40/100
    for i in range(pop_out):
        for j in range(wl):
            if np.random.rand()<chance:
                population[i][j]+=random.normalvariate(0,d)

def eliminate():
    global population
    os=np.zeros(pop_out)
    for i in range(pop_out):
        os[i]=o(population[i])
    global best
    best.append(sorted(os)[:])
    global bestw
    bestw=population[os.argsort()[-10:]]
    os[os.argsort().argsort()>=pop_out-4]*=100
    tmpPopu=np.zeros((pop_out,wl))
    osSum=np.sum(os)
    os[:]/=osSum
    indx=np.random.choice(range(pop_out),pop_in,replace=False,p=os)
    for i,ind in enumerate(indx):
        tmpPopu[i]=population[ind]
    population=tmpPopu

def nrecommend0(original,num):
    return recommend(np.ones(1028),original,num)

def nrecommend1(original,num):
    w=np.zeros(1028)
    for i in range(16):
        w[i]=np.random.randint(0,2)
    return recommend(w,original,num)

def nrecommend2(original,num):
    w=[2**(16-i) for i in range(32)]
    return recommend(w,original,num)


import datetime
import time
random.seed(0x33ba87cd)resolution=10
qs=[i/resolution for i in range(0,resolution+1,1)]
batches=12
iters=25
print(qs)
bestws=[[] for _ in range(len(qs))]
bestOs=[[] for _ in range(len(qs))]
start=time.time()
for qii,qi in enumerate(qs):
    global q,best
    best=[]
    q=qi
    for cnt in range(batches):
        initGA()
        for i in range(iters):
            parents=selectParents()
            crossover_2(parents)
            mutate()
            eliminate()
            c,ci=obtainc(cnum,csize,np.random.randint(0,2**16))
            
        bestOs[qii].append(best)
        best=[]
        bestws[qii].extend(bestw)
        prog=(qii*batches+cnt+1)/(len(qs)*batches)
        stop=time.time()
        print("q: "+str(qi)+"; batch: "+str(cnt)+" || progress: "+f'{prog*100:.3f}'+"% || Estimated time: "+str(datetime.timedelta(seconds = (stop-start)/prog - (stop-start))))


from itertools import chain
txx=[[i]*15 for i in range(iters)]
tx=[txx for _ in range(batches)]
t0=list(chain.from_iterable(tx))
t1t=[[j[:15] for j in i] for i in bestOs[-1:]]
t1=list(chain.from_iterable(t1t))



plt.scatter(t0,t1)
plt.xlabel("q value")
plt.ylabel("f")
plt.show()


import pickle
with open('_bestwstNC3PCA64MODNewDis_.pkl', 'wb') as file: 
    pickle.dump(bestws,file)


c,ci=obtainc(250,8,0xA6768e74)

oss=[]
ods=[]
osts=[]
odts=[]
sa=[]
sd=[]
for wi in bestws:
    ost=0
    odt=0
    osts=[]
    odts=[]
    for wii in wi:
        s=os(wii)
        t=od(wii)
        osts.append(s)
        odts.append(t)
        ost+=s
        odt+=t
    sa.append(osts)
    sd.append(odts)
    oss.append(ost/len(wi))
    ods.append(odt/len(wi))

ys=[]
for i,v in enumerate(qs[:-1]):
    for j in range(len(sa[i])):
        ys.append(v)


plt.figure(figsize=(16,9))

plt.boxplot(sa)
plt.xlabel("q value")
plt.ylabel("similarity")


plt.show()

plt.scatter(ys,sd[:-1])
plt.xlabel("q value")
plt.ylabel("Dissimilarity")
plt.show()

plt.scatter(ys,sa[:-1])
plt.xlabel("q value")
plt.ylabel("Dissimilarity")
plt.show()

x=qs[:]
k1=oss
plt.plot(x,k1)
plt.xlabel("q value")
plt.ylabel("Similarity")
plt.show()

x=qs[:]
k1=ods
plt.plot(x,k1)
plt.xlabel("q value")
plt.ylabel("Dissimilarity")
plt.show()

def centering(o,ps):
    ocenter=np.average(o,axis=0)
    return [i-ocenter for i in ps]

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(XA)
XAD=pca.transform(XA)
pcanum=4
tc,tci=obtainc(pcanum,1,0xe638cccc)


import matplotlib.cm as cm
tr=[[recommend(j,i,5) for j in bestws[0]] for i in tc]
tciduped=[]
for i in tci:
    tciduped.extend([i]*len(bestws[0]))
pcapts=[centering(XAD[tci[i]],XAD[tr[i]]) for i in range(len(tr))]
plt.figure(figsize=(9,9))
colors = cm.autumn(np.linspace(0, 1, len(pcapts)))
for i,c in zip(pcapts,colors):
    tx=[j[0] for j in i]
    ty=[j[1] for j in i]
    plt.scatter(tx,ty,color=c)

tr=[[recommend(j,i,5) for j in bestws[17]] for i in tc]
tciduped=[]
for i in tci:
    tciduped.extend([i]*len(bestws[17]))
pcapts=[centering(XAD[tci[i]],XAD[tr[i]]) for i in range(len(tr))]
colors = cm.winter(np.linspace(0, 1, len(pcapts)))
for i,c in zip(pcapts,colors):
    tx=[j[0] for j in i]
    ty=[j[1] for j in i]
    plt.scatter(tx,ty,color=c)

plt.grid(True)
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)

plt.boxplot([d0,d1])
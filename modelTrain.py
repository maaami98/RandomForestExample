from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

file_path = "payloads.csv"
# benim verilerim örnek
"""
"payload","length","attack_type","label"
"40184","5","norm","norm"
"""
# verilerimi okuttum
df = pd.read_csv(file_path, usecols=[1,2,3], delimiter=",", names=['payload','is_malicious','injection_type'], encoding='utf-8')
# okunan veri tiplerinin durumunu ekrana bastırma

df['payload'] = df['payload'].astype(str)
df['injection_type'] = df['injection_type'].astype(str)
df['is_malicious'] = df['is_malicious'].astype(str)
df.info()
print (df.head(30))

#verilerimizi vektörlere dönüştürüyoruz
vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3))

# payload verimizi x vektörü yapıyoruz
tfidf = vectorizer.fit(df.payload)

X=vectorizer.fit_transform(df.payload)
# injection_type verimizi çıktı olan y vektörü yapıyoruz
y=df.injection_type

#Veriyi eğitim ve test alt-veri setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) # 70% Öğrenme and 30% test

clf=RandomForestClassifier(n_estimators=100) #Karar ağacı modeli oluşturma
clf.fit(X_train,y_train) #Modeli eğitim verisine sadeleştirme 


y_pred=clf.predict(X_test)
print("Doğruluk:",metrics.accuracy_score(y_test, y_pred))

pickle.dump(clf, open("model.pickle", 'wb'))
pickle.dump(tfidf, open("vectorizer.pickle", "wb"))
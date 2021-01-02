import pickle

model_path = "model.pickle"
vectorizer_path = "vectorizer.pickle"

vectorizer = pickle.load(open(vectorizer_path,'rb'))
model = pickle.load(open(model_path,'rb'))


while True:
	payload = input("Giriş Girin:")
	sonuc = model.predict(vectorizer.transform([str(payload)]))
	print (sonuc)
	

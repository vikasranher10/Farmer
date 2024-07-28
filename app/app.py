
from flask import Flask, render_template, request, Markup , redirect
import numpy as np
import pandas as pd

from utils.fertilizer import fertilizer_dic

import sklearn
import pickle
import os
# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__)



@ app.route('/')
def home():
    title = 'Farmer - Home'
    return render_template('index.html', title=title)




@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Farmer - Crop Recommendation'
    return render_template('rcrop.html', title=title)






@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Farmer - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)



@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice(तांदूळ)", 2: "Maize(मका)", 3: "Jute(ज्यूट)", 4: "Cotton(कापूस)", 5: "Coconut(नारळ)", 6: "Papaya(पपई)", 7: "Orange(संत्रा)",
                 8: "Apple(सफरचंद)", 9: "Muskmelon(कस्तुरी)", 10: "Watermelon(टरबूज)", 11: "Grapes(द्राक्षे)", 12: "Mango(आंबा)", 13: "Banana(केळी)",
                 14: "Pomegranate(डाळिंब)", 15: "Lentil(मसूर)", 16: "Blackgram(ब्लॅकग्राम)", 17: "Mungbean(मूग)", 18: "Mothbeans(मोथबीन्स)",
                 19: "Pigeonpeas(कबुतर)", 20: "Kidneybeans(राजमा)", 21: "Chickpea(चणे)", 22: "Coffee(कॉफी)"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} हे लागवडीसाठी सर्वोत्तम पीक आहे".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('rcrop.html',result = result)




@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Farmer - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)




if __name__ == '__main__':
    app.run(debug=True)

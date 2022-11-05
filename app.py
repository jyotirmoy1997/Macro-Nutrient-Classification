import pickle
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET']) # This is the root route
def root():
    if request.method == "POST":
        ckb = int(request.form.get("ckb"))
        ph = float(request.form.get("pH"))
        OC = float(request.form.get("OC"))
        EC = float(request.form.get("EC"))
        S = float(request.form.get("S"))
        Zn = float(request.form.get("Zn"))
        Fe = float(request.form.get("Fe"))
        Cu = float(request.form.get("Cu"))
        Mn = float(request.form.get("Mn"))
        B = float(request.form.get("B"))

        # For Nitrogen

        if ckb == 1:
            P = float(request.form.get("P"))
            K = float(request.form.get("S"))
            N_Model = pickle.load(open('HardVotingClassifierModel_Nitrogen.pkl', 'rb'))
            N_test_li = np.array([ph, OC, EC, P, K, S, Zn, Fe, Cu, Mn, B]).reshape(1, -1)
            res = N_Model.predict(N_test_li)
            
            result = ""
            if res[0] == 1:
                result = "Very Low"
            elif res[0] == 2:
                result = "Low"
            elif res[0] == 3:
                result = "Medium"
            elif res[0] == 4:
                result = "High"
            elif res[0] == 5:
                result = "Very High"
            
            return render_template('result.html', val=result)
        # For Potassium

        elif ckb == 2:
            P = float(request.form.get("P"))
            N = float(request.form.get("N"))
            K_Model = pickle.load(open('HardVotingClassifierModel_Potassium.pkl', 'rb'))
            K_test_li = np.array([ph, OC, EC, P, N, S, Zn, Fe, Cu, Mn, B]).reshape(1, -1)
            res = K_Model.predict(K_test_li)
            
            result = ""
            if res[0] == 1:
                result = "Very Low"
            elif res[0] == 2:
                result = "Low"
            elif res[0] == 3:
                result = "Medium"
            elif res[0] == 4:
                result = "High"
            elif res[0] == 5:
                result = "Very High"

            return render_template('result.html', val=result)

        elif ckb == 3:
            N = float(request.form.get("N"))
            K = float(request.form.get("K"))
            P_Model = pickle.load(open('SoftVotingClassifierModel_Phosphorus.pkl', 'rb'))
            P_test_li = np.array([ph, OC, EC, K, N, S, Zn, Fe, Cu, Mn, B]).reshape(1, -1)
            res = P_Model.predict(P_test_li)
            
            result = ""
            if res[0] == 1:
                result = "Very Low"
            elif res[0] == 2:
                result = "Low"
            elif res[0] == 3:
                result = "Medium"
            elif res[0] == 4:
                result = "High"
            elif res[0] == 5:
                result = "Very High"

            return render_template('result.html', val=result)
        
    return render_template('index.html') # render_template sends the HTML file to the browser


@app.route('/about.html')
def about():
    return render_template('about.html')

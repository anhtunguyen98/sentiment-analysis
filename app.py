from flask import Flask,render_template,request,json
from infer import *
app=Flask(__name__)
lstm_model=LSTM_Model()
cnn_model=CNN_Model()
@app.route('/', methods=['POST','GET'] )
def welcome():

    return render_template('index.html')

@app.route('/process',methods=['POST'])
def predict():
    text = request.get_json()['text']
    type=request.get_json()['type']
    if type=='lstm':

        negative,positive=lstm_model.result(text)
    else:
        negative, positive = cnn_model.result(text)
    if negative>positive:
        result={'result':'Negative','negative':str(negative),'positive':str(positive)}
    else:
        result = {'result': 'Positive', 'negative': str(negative), 'positive': str(positive)}
    return json.dumps({'success': True, 'result': result}), 200, {'Content-Type': 'application/json; charset=UTF-8'}

if __name__=='__main__':

    app.run(host='0.0.0.0',port=5000,debug=False)
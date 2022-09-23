from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from flask import Flask, request, render_template
#import json

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

app = Flask(__name__)

@app.route("/")
def home():
    return 'Welcome to Home Page of Tokenization. Click <a href="/inputstr">here</a> to explore.'

@app.route('/inputstr',methods = ['POST', 'GET'])
def inputstr():
    if request.method == 'POST':
        textstr = request.form['strval']
        output = nlp(textstr)
        for js in output:
            # For debugging
            print('Word:', js['word'], ', Token Type:', js['entity_group'])
        return render_template('inputstr.html', textstr='Original String: '+textstr, result='Output: '+str(output))
    else:
        return render_template('inputstr.html', textstr='Enter String Above', result='To Tokenize it')


@app.route("/test")
def ping():
    return 'Test Page'

if __name__ == '__main__':
    #Initialize Tokenizer
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    app.run(debug=True, host='0.0.0.0', port=8080) #Run Flask App on port 8080 instead of default 5000
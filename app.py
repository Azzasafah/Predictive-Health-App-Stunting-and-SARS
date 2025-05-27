from flask import Flask, render_template, request
from controllers.stunting_controller import predict_stunting
from controllers.sars_controller import predict_sars

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/sars', methods=['GET', 'POST'])
def sars():
    hasil = None
    if request.method == 'POST':
        hasil = predict_sars(request.form)
    return render_template('sars.html', hasil=hasil)

@app.route('/stunting', methods=['GET', 'POST'])
def stunting():
    hasil = None
    if request.method == 'POST':
        hasil = predict_stunting(request.form)
    return render_template('stunting.html', hasil=hasil)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, abort, Response
from model import Model
import logging

app = Flask(__name__, template_folder=".")

@app.route('/', methods=["GET"])
def landing():
	return render_template("front.html")

@app.route('/analysis', methods=["POST"])
def generator():
	mdl = Model()
	stim, seq, valid = mdl.generate(request.form.get('fweb'), int(request.form.get('num')), 2)
	if (not valid):
		raise abort(Response("CUSTOM EXCEPTION 400: Invalid Stimulus Data Provided to Model"))
	return render_template("result.html", news=request.form.get('fweb'), stim=stim, seq=seq)

if __name__=="__main__":
	app.run(port=8889)

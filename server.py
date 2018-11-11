from flask import Flask, render_template
import datetime
import pickle

app = Flask(__name__)
@app.route("/")
def get_all_users():
    template_data = {'name':[], 'time':""}
    names = pickle.load(open("names.pkl", "rb"))
    now = datetime.datetime.now()
    
    time_string = now.strftime("%d-%m-%Y %H:%M")
    
    names_list = []
    for name in names:
        x = name.split("_")
        names_list.append(x[0] + " " + x[1])
    
    return render_template('main.html', names_list = names_list, time_string = time_string)

if __name__ == '__main__':
    app.run(host="192.168.0.103", port=8080, debug=True)
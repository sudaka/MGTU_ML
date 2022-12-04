from app import app
from flask import render_template
from flask import request
from datetime import datetime
from app.static import rivermain

def checkinput(year, month, day):
    try:
        cyear = int(year)
        cmonth = int(month)
        cday = int(day)
        datetime.strptime(f'{day}-{month}-{year}', "%d-%m-%Y")
    except:
        cyear = 2011
        cmonth = 1
        cday = 1
    return (cyear, cmonth, cday)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def predict():
    curyear = request.args.get('year')
    curmonth = request.args.get('month')
    curday = request.args.get('day')
    (curyear, curmonth, curday) = checkinput(curyear, curmonth, curday)
    rivermain.minprocedure(curyear, curmonth, curday)
    return render_template("predict.html", year=curyear, month=curmonth, day=curday)
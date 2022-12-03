from app import app
from flask import render_template
from flask import request
from datetime import datetime
from app.static import rivermain

def checkinput(year, month, day):
    if year not in [x for x in range(2011, 2016, 1)]:
        cyear = 2011
    if month not in [x for x in range(1, 13, 1)]:
        cmonth = 1
    if day not in [x for x in range(1, 31, 1)]:
        cday = 1
    try:
        datetime.strptime(f'{day}/{month}/{year}', "%d/%m/%Y")
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
    #(curyear, curmonth, curday) = checkinput(curyear, curmonth, curday)
    rivermain.minprocedure(curyear, curmonth, curday)
    return render_template("predict.html", year=curyear, month=curmonth, day=curday)
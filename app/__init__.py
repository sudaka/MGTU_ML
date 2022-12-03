from flask import Flask

app = Flask(__name__)
from app import views
from app.static import rivermain
from app.static import rivercommon
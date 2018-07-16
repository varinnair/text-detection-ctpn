import os
from flask import Flask, request, redirect, url_for,flash,render_template,Response
from werkzeug.utils import secure_filename
import subprocess
from difflib import get_close_matches
z
UPLOAD_FOLDER = '/home/aditya/aditya/text-detection-ctpn/ctpn/data/demo'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from shutil import copyfile

import time

class tunnig():

    def __init__(self):
        self.text = text

    def conv_list(self):
        with open('data.txt', 'r') as myfile:
            data = myfile.read().replace('\n', '')

    def closeMatches(self, word):
        word = ''.join(e for e in word if e.isalnum())
        word.lower()
        print(word)
        a = get_close_matches(word, self.pattern)
        #holla.append(a)
        return a

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            global filename
            filename = secure_filename(file.filename)
            print(filename)
            try:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg'))
                copyfile(os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg'),'/home/aditya/aditya/text-detection-ctpn/ctpn/data/img1/img1.jpg')
            except Exception as e:
                print(e)
            try:
                process = subprocess.Popen(["python", "demo.py"],
                                            stdout=subprocess.PIPE)
                stdout = process.communicate()[0].decode("utf-8").split()
                time.sleep(10)
            except:
                print("error")
            a = time.time()
            copyfile(os.path.join(app.config['UPLOAD_FOLDER'],'img1.jpg'),'/home/aditya/aditya/text-detection-ctpn/data'+str(a)+'.jpg')
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],'img1.jpg'))
            copyfile('/home/aditya/aditya/text-detection-ctpn/ctpn/data/1.txt','/home/aditya/aditya/text-detection-ctpn/ctpn/data/results/' +str(a)+'.txt')

            global f
            import re

            fileName1 = '/home/aditya/aditya/text-detection-ctpn/ctpn/data/1.txt'
            file1 = open(fileName1, 'r').read().split('\n')
            file1 = [element.lower() for element in file1]

            fileName2 = '/home/aditya/aditya/text-detection-ctpn/ctpn/testing.txt'
            file2 = open(fileName2, 'r').read().split('\n')
            file2 = [element.lower() for element in file2]

            c =[]
            chars = '[\n\t\r©|{}@#%&*()+-/\<>?$^!]'
            for i in file1:
                new_text = re.sub(chars, '', i)
                c.append(new_text)

            d = []
            for i in c:
                for word in i.split():
                    if word in file2:
                        d.append(word)
            print(d)
            os.remove('/home/aditya/aditya/text-detection-ctpn/ctpn/data/1.txt')
            return render_template('display.html',d = d)
    else:
        return render_template('upload.html')

if __name__ == '__main__':
     app.run(debug = 'True')
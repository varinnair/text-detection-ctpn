#subprocess.run(["curl", i, "--output", str(new.id) + ".jpg"])
import subprocess
#a = subprocess.run(["gocr","--output","1x.jpg"])
#from subprocess import check_output
#out = check_output(["gocr","1x.jpg"])
holla = []
a11 = []
from difflib import get_close_matches

path = 'subsection_10.jpg'

def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

class gocr():
    def __init__(self, path):
        self.path = str(path)
        self.pattern = []

    def change_path(self, new_path):
        self.path = str(new_path)

    def add_pattern(self,line):
        self.pattern.append(line)

    def get_path(self):
        return self.path

    def pattern1(self):
        text_file = open("testing.txt", "r")
        lines = text_file.readlines()
        for line in lines:
            self.add_pattern(line.strip())
            #a11.append(line.strip())
        text_file.close()

    def closeMatches(self, word):
        word = ''.join(e for e in word if e.isalnum())
        print(word)
        a = get_close_matches(word.lower(),self.pattern)
        holla.append(a)
        return a

    def process(self):
        self.pattern1()
        self.change_path(path)
        process = subprocess.Popen(['gocr', str(self.path)], stdout=subprocess.PIPE)
        stdout = process.communicate()[0].decode("utf-8").split()
        print(stdout)
        for word in stdout:
            a = self.closeMatches(word)
            holla.append(a)
        holla1 = [x[0] for x in holla if x]
        holla1 = Remove(holla1)
        #holla1 = [x[0] for x in holla1]
        #holla1 = set(holla1)
        return holla1

#holla1 = [x for x in holla if x]
#holla1 = [x[0] for x in holla1]
#print(holla1)
#print(holla1)

h1 = gocr(path)
a2 = h1.process()

print(a2)




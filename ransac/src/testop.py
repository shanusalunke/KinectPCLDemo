import numpy
import sys
from os import listdir
from os.path import isfile, join
from pprint import pprint
import json

def getlst(str):
    str = str.replace("(",",").replace(")",",").replace(" - ",",").replace(" ","")
    lst = str.split(",")[1:-1]
    intlst = [float(i) for i in lst]
    # return numpy.array(intlst)
    return intlst

def printCommands():
    comm = ["normals", "area", "volume", "normals2", "fpfhs"] #gradient
    for c in comm:
        print c

def printConfusion():

    print "\t",
    for col in allvar:
        print col, "\t",
    print ""
    for i in range(0,len(allvar)):
        row = mat[i]
        print allvar[i], "\t",
        for val in row:
            print "%.2f" % val, "\t",
        print "\n"
    print "\t",
    for col in allvar:
        print col, "\t",
    print ""

def printLinear():
    for col in allvar:
        print col, "\t",
    print ""
    for a in normval:
        print "%.2f" % a["val"], "\t",
    print ""

def plotLinear():
    import pylab
    # from pylab import *
    pylab.ylim([0,200])
    pylab.ylim([0,4])
    for k in normval:
        # print k["key"], k["val"]
        pylab.plot(k["val"], k["key"], '+')
    pylab.show()


# print sys.argv
point = sys.argv[1]

if point == "?":
    printCommands()
    sys.exit(0)

#Read files
mypath = 'json'
onlyfiles = sorted([ f for f in listdir(mypath) if isfile(join(mypath,f))])
print "Files found: ", onlyfiles

data = {}
for f in onlyfiles:
    with open(join(mypath,f)) as data_file:
        d = json.load(data_file)
        data[f.split('.')[0]] = d

allvar = sorted(data.keys())
mat = []
normval = [] #list of key-value pairs for plotting
normval2 = [] #list of values only
# print allvar
for i in allvar:
    row = []
    a = numpy.array(getlst(data[i][point])) if isinstance(data[i][point],basestring) else data[i][point]
    k = 0
    if i[0] == 'C':
        k = 1 + 0.1*int(i[1])
    elif i[0] == 'B':
        k = 2+ 0.1*int(i[1])
    else:
        k = 3+ 0.1*int(i[1])
    print k, " ", i[0], numpy.linalg.norm(a)
    normval.append({"key":k,"val":numpy.linalg.norm(a)})
    normval2.append(numpy.linalg.norm(a))
    for j in allvar:
        b = numpy.array(getlst(data[j][point])) if isinstance(data[j][point],basestring) else data[j][point]
        row.append(numpy.linalg.norm(a-b))
    mat.append(row)


mat2 = []
for i in allvar:
    a = getlst(data[i][point]) if isinstance(data[i][point],basestring) else data[i][point]
    # print a
    mat2.append(a)

print mat2

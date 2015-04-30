import csv

class ObjectAppearance:

    def __init__(self, category_label, nx, ny, nz, nd, area, volume, mass, friction_coeff):
        self.category_label = category_label
        self.nx = float(nx)
        self.ny = float(ny)
        self.nz = float(nz)
        self.nd = float(nd)
        self.area = float(area)
        self.volume = float(volume)

        self.mass = float(mass)
        self.friction_coeff = float(friction_coeff)

        #derived
        if self.volume > 0:
            self.density = self.mass/self.volume
        else:
            self.density = 0

    def __repr__(self):
        return self.category_label


def getObjectAppearances(filename):

    listOfObjects = []

    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        spamreader.next() #ignore first line
        for row in spamreader:
            listOfObjects.append(ObjectAppearance(row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]))

    return listOfObjects

import sys
a = sys.argv[1]
b = sys.argv[2]
c = sys.argv[3]

def saludar(par1, par2, par3):
    return("Hola desde python, " + str(par1) + ", " + str(par2) + " " + str(par3))

print(saludar(a, b, c))
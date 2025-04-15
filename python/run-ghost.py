import sys
from Ghost import *
if __name__ == "__main__":
    g = Ghost('.')
    g.invoke(sys.argv[1])
    sys.exit(0)

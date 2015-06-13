import sys

def convert_dimac_2_csr(readfile, writefile):
    with open(readfile, "r") as rfile:
        with open(writefile, "w") as wfile:
            # read cluster lines
            for graph_line in rfile:
                if "c " in graph_line:
                    continue
                if "p " in graph_line:
                    continue
                if "a " in graph_line:
                    temp = graph_line.strip("a ").replace(" ", ",")
                    if "\n" not in temp:
                        temp += '\n'
                    wfile.writelines(temp)

def main(argv):

    if len(argv) <= 2:
    	print argv
        print "Usage: python dimac2csr in_file out_file"
        sys.exit()

    readfile = argv[1]
    writefile = argv[2]
    convert_dimac_2_csr(readfile, writefile)

    print "finished converting"

if __name__ == "__main__":
    main(sys.argv)
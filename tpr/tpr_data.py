with open(r"..\data\train10pc", 'r') as file:
    # for i, l in enumerate(file):
    #     pass
    # print(i + 1)
    for i in range(10):
        line = file.readline()
        print(line)
    # line_splited = line.split(',')
    # print(len(line_splited))
    # for item in line_splited:
    #     print(item)

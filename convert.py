import re


def main():
    f = open('data/brilliant2/array4.txt')
    f1 = open('data/brilliant2/array5.txt', 'w')
    a=[]
    for line in f:
        ind = line.find(',')
        line2 = line[ind+1:len(line)]
        a.append(line2)

    with f1 as fw:
        for item in a:
            fw.write(item)
        # записываем


if __name__ == '__main__':
    main()
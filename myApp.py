import csv
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

#out = np.zeros(512)

#with open('data/dir/1.csv', newline='') as csvfile:
 #   spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
  #  i = 0
 #   for row in spamreader:
  #      print(row)
  #      out[i] = row[1]
  #      i += 1

#for rowa in out:
#np.savetxt(a_file, out)

#a_file.close()
#with open("output.txt", "w") as txt_file:
    #for line in out:
       # print("".format(line), file=txt_file)
#np.savetxt("data/arzray.txt", out, fmt="%s")
f = open("data/sin(x+1).txt", "r")
data = np.loadtxt(f, delimiter='\n', dtype=np.float)
f.close()


x = np.zeros(2048)
for i in range(2048):
    x[i] = i
ax.plot(x,data)
plt.show()

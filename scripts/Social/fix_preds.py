#!/bin/python
with open("predictions.txt") as pfd, open("predictions2.txt","w+") as ofd:
	for line in pfd:
		ofd.write(str(int(line.strip())+1)+"\n")


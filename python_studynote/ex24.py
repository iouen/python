# -*- coding:utf-8 -*-.
from sys import argv
from os.path import exists#add ex17

script,filename,to_file = argv#add ex17 to_file

print "We're going to erase %r." % filename
print "If you don't want that,hit ctrl-c(^C)."
print "If you do want that,hit RETURN."
raw_input("?")
print "Opening the file ...."
target = open(filename,'w')
print "Truncating the file.Goodbye!"
target.truncate()

print "Now I'm going to ask you for three lines"
line1=raw_input("line1:")
line2=raw_input("line2:")
line3=raw_input("line3:")

print "I'm going to write these to the file."

target.write(line1)
target.write("\n")
target.write(line2)
target.write("\n")
target.write(line3)
target.write("\n")

print "And finally,we close it."
target.close()

print "Start ---------copy"
in_file = open(filename)
indata = in_file.read()
print "The input file is %d bytes long"%len(indata)
print "Does the output file exists?%r" % exists (to_file)
print "Ready,hit RETURN to continue,CTRL-C to abort."
raw_input()
out_file = open(to_file,'w')
out_file.write(indata)
print "Alright , all done."
out_file.close()
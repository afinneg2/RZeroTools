#!/usr/bin/env python

import sys
import time

def main():
    sys.stdout.write("8==>")
    sys.stdout.flush()


    for i in range(10):
        time.sleep(0.2)
        sys.stdout.write("\b\b")
        sys.stdout.write("==>")
        sys.stdout.flush()

    for i in range(10):
        time.sleep(0.2)
        sys.stdout.write("\b\b")
        sys.stdout.write(">")
        sys.stdout.flush()
        time.sleep(0.2)
        sys.stdout.write("\b")
        sys.stdout.write("=>")
        sys.stdout.flush()

    for i in range(5):
        sys.stdout.write("-")
        sys.stdout.flush()
        time.sleep(0.2)
        
    sys.stdout.write("X")
    sys.stdout.flush()
    time.sleep(0.2)
    sys.stdout.write("\b"*6)
    sys.stdout.write(" "*5 + "|")
    sys.stdout.flush()
    time.sleep(0.2)
    sys.stdout.write("\b")
    sys.stdout.write( "_")
    sys.stdout.flush()
    time.sleep(1)
    sys.stdout.write("\n")
    return None



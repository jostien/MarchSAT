#!/bin/bash
g++ -Wall -g -I/home/jostie/tools/AMDAPPSDK-3.0/include -L/home/jostie/tools/AMDAPPSDK-3.0/lib/x86_64 main.cc -lOpenCL -lboost_program_options

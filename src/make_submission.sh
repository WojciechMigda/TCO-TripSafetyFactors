#!/bin/sh

cat num.hpp sigmoid.hpp fmincg.hpp array2d.hpp logreg.hpp TripSafetyFactors.hpp | grep -v "#include \"" > submission.cpp
g++ -std=c++11 -c submission.cpp
gvim submission.cpp &

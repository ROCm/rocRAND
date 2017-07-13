#!/bin/bash

gnuplot --persist -d -e "plot '$1' using 1 title 'observed' with steps, '$1' using 2 title 'expected' with steps ; pause mouse close"

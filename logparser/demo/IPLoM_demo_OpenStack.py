# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:17:48 2021

@author: paris
"""

import sys
sys.path.append('../')
from logparser import IPLoM

input_dir    = '/home/terminator18ops/Documents/Aditya/SEM-5/cloud_computing/Project/OpenStack/openstack-logs/'  # The input directory of log file
output_dir   = '20k+4k destroy+10k undefined +4k dhcp_result/'  # The output directory of parsing results
log_file     = 'openstack-nova-sample.log'  # The input log file name
log_format   = '<LevelName> <name> <request_id> <Content>'  # OpenStack log format
maxEventLen  = 500  # The maximal token number of log messages (default: 200)
step2Support = 0  # The minimal support for creating a new partition (default: 0)
CT           = 0.9  # The cluster goodness threshold (default: 0.35)
lowerBound   = 0.25  # The lower bound distance (default: 0.25)
upperBound   = 0.9  # The upper bound distance (default: 0.9)
regex        = [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\[instance: \w*-\w*-\w*-\w*-\w*\]']  # Regular expression list for optional preprocessing (default: [])

parser = IPLoM.LogParser(log_format=log_format, indir=input_dir, outdir=output_dir,
                         maxEventLen=maxEventLen, step2Support=step2Support, CT=CT, 
                         lowerBound=lowerBound, upperBound=upperBound, rex=regex)
parser.parse(log_file)

import numpy as np 
import os, time
import pandas as pd
import subprocess

def get_points_number(filedir):
    plyfile = open(filedir)

    line = plyfile.readline()
    while line.find("element vertex") == -1:
        line = plyfile.readline()
    number = int(line.split(' ')[-1][:-1])
    
    return number

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item) 
        except ValueError:
            continue
        
    return number

def pc_error(infile1, infile2, normal1, res, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    # Use A as reference
    headers1 = ["h.       1(p2point)", "h.,PSNR  1(p2point)", \
               "h.       1(p2plane)", "h.,PSNR  1(p2plane)", \
               "mse1      (p2point)", "mse1,PSNR (p2point)", \
               "mse1      (p2plane)", "mse1,PSNR (p2plane)"]
    
    # Use A as reference
    headers2 = ["h.       2(p2point)", "h.,PSNR  2(p2point)", \
               "h.       2(p2plane)", "h.,PSNR  2(p2plane)", \
               "mse2      (p2point)", "mse2,PSNR (p2point)", \
               "mse2      (p2plane)", "mse2,PSNR (p2plane)"]

    # Symmetric Metrics.
    headers3 = ["h.        (p2point)", "h.,PSNR   (p2point)", \
               "h.        (p2plane)", "h.,PSNR   (p2plane)", \
               "mseF      (p2point)", "mseF,PSNR (p2point)", \
               "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headers3

    command = str('myutils/pc_error_d' + 
                          ' -a '+infile1+ 
                          ' -b '+infile2+ 
                          ' -n '+normal1+
                          ' --hausdorff=1 '+ 
                          ' --resolution='+str(res-1))

    results = {}
   
    start = time.time()
    subp=subprocess.Popen(command, 
                          shell=True, stdout=subprocess.PIPE)

    c=subp.stdout.readline()
    while c:
        # line = c.decode(encoding='utf-8')# python3.
        line = str(c, encoding="utf8") # python2.
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c=subp.stdout.readline() 
    print('===== measure PCC quality using `pc_error` version 0.13.4', round(time.time() - start, 4))

    return pd.DataFrame([results])

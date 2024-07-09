import os
import json
import pandas as pd
import numpy as np
import math
import requests
from tqdm import tqdm
from multiprocessing import Process

# functions


def download_img(url):
    fname = url.split('/')[-1]
    save_path = "./dataset/raw"
    full_path = os.path.join(save_path, fname)
    existed = os.path.exists(full_path)
    if(existed == False):
        # get the file content
        response = requests.get(url)
        # save itgi
        open(full_path, "wb").write(response.content)
    return None


def multi_download(part_data):
    for mem in tqdm(part_data):
        # default variables
        id_ = ""
        image_url = ""

        # assign variables
        id_ = mem['_id']['$oid']
        image_url = mem['image_url']

        # Download images
        download_img(image_url)
    return None


def cut_data(full, num):
    dem = 1
    nlns = len(full)
    step = nlns / num
    print("step =", step)
    st = 0
    ed = math.floor(st + dem*step)
    kq = []
    while (ed <= nlns):
        print(f"part = {dem}, from {st} to {ed}")
        part = full[st:ed]
        kq.append(part)
        dem += 1
        st = ed
        ed = math.floor(dem*step)

    return kq


if __name__ == '__main__':
    # read json file to a list
    refined = []
    data = []
    with open('./dataset/pizzacam.json') as f:
        for line in f:
            data.append(json.loads(line))
    # Refine data, some image_url are Null
    for mem in tqdm(data):
        if (mem['image_url'] != None):
            refined.append(mem)

    # number of lines in dataset
    num_line = len(refined)
    print('Number of lines =', num_line)
    num_cpu = os. cpu_count()
    print("Num process = ", num_cpu)

    # Step 1, cut data set into part
    ls_parts = cut_data(refined, num_cpu)

    procs = []
    for part in ls_parts:
        proc = Process(target=multi_download, args=(part,))
        procs.append(proc)
        proc.start()
        print(f"Process = {proc.pid} started")

    # complete the processes
    for proc in procs:
        proc.join()
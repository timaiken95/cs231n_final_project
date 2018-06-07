import numpy as np
import requests
import threading
import re

with open("dpchallenge_dataset.txt") as f:
    content = f.readlines()

regex = r'<img\ssrc=\"(http://images.dpchallenge.com/[^\"]*)\"\swidth=\"\d+\"\sheight=\"\d+\"'

to_service = []
threadLock = threading.Lock()
threadSem = threading.Semaphore(0)
urls = []
failed_urls = []

def process_data():
    while True:
        threadSem.acquire()
        threadLock.acquire()
        i, url_to_request = to_service.pop(0)
        threadLock.release()
        
        if i == -1:
            break
        
        if i % 100 == 0:
            print(i)
            
        if i % 1000 == 0:
            np.save("photo_urls.npy", urls)
        
        try:
            r = requests.get(url_to_request, headers={'User-Agent': 'Chrome'})
        except:
            threadLock.acquire()
            failed_urls.append((i, url_to_request))
            threadLock.release()
            continue
            
        image_url = re.findall(regex, str(r.text))
        print(r.text)
        
        if image_url:
            threadLock.acquire()
            urls.append((i, image_url))
            threadLock.release()

threads = []
numThreads = 4
for i in range(numThreads):
    t = threading.Thread(target=process_data)
    threads.append(t)
    t.start()            

for i, line in enumerate(content):
    tokens = line.rstrip().split()
    url = "http://www.dpchallenge.com/image.php?IMAGE_ID=" + tokens[1]
    
    threadLock.acquire()
    to_service.append((i, url))
    threadLock.release()
    threadSem.release()
    
for _ in range(numThreads):
    threadLock.acquire()
    to_service.append((-1, ""))
    threadLock.release()
    threadSem.release()

for t in threads:
    t.join()

np.save("photo_urls.npy", urls)
np.save("failed_urls.npy", failed_urls)
import numpy as np
import requests
import threading
import re

with open("full_imageurls.txt") as f:
    content = f.readlines()

regex = r'(\d+)\s\[\'(http://images.dpchallenge.com/images_challenge[^\']+)\''

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
        
        try:
            image_data = requests.get(url_to_request)
            with open("image" + str(i) + ".jpg", 'wb') as handler:
                handler.write(image_data)

        except:
            continue

threads = []
numThreads = 8
for i in range(numThreads):
    t = threading.Thread(target=process_data)
    threads.append(t)
    t.start()            

for line in content:

    matches = re.findall(regex, line)
    if len(matches) > 1:
        threadLock.acquire()
        to_service.append((matches[0][0], matches[0][1]))
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
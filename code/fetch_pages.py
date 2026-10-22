import re, time, os, sys, requests
from concurrent.futures import ThreadPoolExecutor
nums=[int(x) for x in re.findall(r'- number: "(\d+)"',open('data/erdosproblems/data/problems.yaml').read())]
s=requests.Session(); s.headers['User-Agent']='ai-math-forecast-research/0.1 (personguy976@gmail.com)'
def get(n):
    p=f'data/pages/{n}.html'
    if os.path.exists(p): return
    for t in range(3):
        try:
            r=s.get(f'https://www.erdosproblems.com/{n}',timeout=30)
            if r.status_code==200: open(p,'w').write(r.text); break
            time.sleep(2+t*3)
        except Exception as e: time.sleep(2+t*3)
    time.sleep(0.3)
with ThreadPoolExecutor(3) as ex: list(ex.map(get,nums))
print('done',len(os.listdir('data/pages')))

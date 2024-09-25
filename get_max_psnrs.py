import json
import os

outputs = os.listdir('output') + os.listdir('output/db') + os.listdir('output/tandt')
table = {}

for i in outputs:
    if '4096' in i or '16384' in i:
        try:
            try:
                data = json.load(open(os.path.join('output', i, 'results.json')))
            except:
                try:
                    data = json.load(open(os.path.join('output/tandt', i, 'results.json')))
                except:
                    data = json.load(open(os.path.join('output/db', i, 'results.json')))
            max_psnr = data['ours_3000']['PSNR']
            print(i, max_psnr)
        except Exception as e:
            print(e)

"""Prints active-block usage as % of the largest previous block (tokens). Exit 0 always; prints PCT."""
import subprocess, json
o=subprocess.run(['npx','-y','ccusage@latest','blocks','--json'],capture_output=True,text=True).stdout
b=json.loads(o)['blocks']
act=[x for x in b if x.get('isActive')]
prev=[x['totalTokens'] for x in b if not x.get('isActive') and not x.get('isGap')]
cur=act[0]['totalTokens'] if act else 0
mx=max(prev+[1])
print(f"PCT={100*cur/mx:.0f} cur_tokens={cur} max_prev_block={mx} cost={act[0].get('costUSD') if act else 0}")

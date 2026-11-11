import json, urllib.request, statistics, datetime, time, random
import urllib.parse
def parse(s):
    for f in ("%d %B %Y","%Y-%m-%d"):
        try: return datetime.datetime.strptime(s.strip(),f)
        except: pass
items=[]; cur="*"
while True:
    url=f"https://api.crossref.org/journals/2213-1248/works?filter=from-pub-date:2025-01-01,type:journal-article&rows=500&select=DOI,assertion,title,published&cursor={urllib.parse.quote(cur)}"
    d=json.load(urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"research-script/1.0"}),timeout=90))["message"]
    if not d["items"]: break
    items+=d["items"]; cur=d["next-cursor"]; time.sleep(1)
import urllib.parse
rows=[]
for w in items:
    r=a=None
    for s in w.get("assertion",[]):
        n=s.get("name","").lower()
        if n=="received": r=parse(s["value"])
        if n=="accepted": a=parse(s["value"])
    if r and a: rows.append((a,(a-r).days,w["DOI"],(w.get("title") or [""])[0]))
print("articles",len(items),"with dates",len(rows))
for lab,lo,hi in [("accepted 2025H1","2025-01-01","2025-07-01"),("accepted 2025H2","2025-07-01","2026-01-01"),("accepted 2026H1","2026-01-01","2026-07-01"),("accepted 2026H2","2026-07-01","2027-01-01")]:
    d=sorted(x[1] for x in rows if lo<=x[0].strftime("%Y-%m-%d")<hi)
    if d: print(f"{lab}: n={len(d)} median={statistics.median(d)} q1={d[len(d)//4]} q3={d[3*len(d)//4]} p90={d[int(.9*len(d))]} frac<=72={sum(v<=72 for v in d)/len(d):.2f} frac<=60={sum(v<=60 for v in d)/len(d):.2f}")
json.dump([(x[0].strftime("%Y-%m-%d"),x[1],x[2],x[3]) for x in rows],open("jksu.json","w"))
random.seed(0)
rec=[x for x in rows if x[0].year==2026]
for x in random.sample(rec,60): print(x[1],"|",x[3][:130])

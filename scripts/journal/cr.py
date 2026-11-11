import sys, json, urllib.request, urllib.parse, statistics, datetime, re, time
def parse(s):
    s=s.strip()
    for f in ("%d %B %Y","%Y-%m-%d","%B %d, %Y","%d %b %Y","%Y-%m-%dT%H:%M:%SZ"):
        try: return datetime.datetime.strptime(s,f)
        except: pass
    return None
def q(issn, since="2026-03-01", rows=200):
    url=f"https://api.crossref.org/journals/{issn}/works?filter=from-pub-date:{since},type:journal-article&rows={rows}&select=DOI,assertion,title,published&sort=published&order=desc"
    for a in range(3):
        try:
            d=json.load(urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"research-script/1.0"}),timeout=60)); break
        except Exception as e:
            time.sleep(3); d=None
    if not d: return None
    it=d["message"]["items"]; days=[]
    for w in it:
        r=a=None
        for s in w.get("assertion",[]):
            n=s.get("name","").lower()
            if n in("received","date_received","received_date"): r=parse(s["value"])
            if n in("accepted","date_accepted","accepted_date"): a=parse(s["value"])
        if r and a and 0<=(a-r).days<1500: days.append((a-r).days)
    return d["message"]["total-results"],len(it),days
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    issn,name=line.split(None,1)
    r=q(issn)
    if not r: print(f"{name}: ERR"); continue
    tot,n,days=r
    if days:
        days.sort(); 
        print(f"{name:55s} total={tot:6d} n={len(days):3d} median={statistics.median(days):5.0f} q1={days[len(days)//4]:4d} q3={days[3*len(days)//4]:4d} frac<=72={sum(x<=72 for x in days)/len(days):.2f}")
    else: print(f"{name:55s} total={tot} sampled={n} no date assertions")

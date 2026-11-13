import sys, urllib.request, urllib.parse, statistics, datetime, re, time, json
E="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
def get(u):
    for a in range(4):
        try: return urllib.request.urlopen(u,timeout=90).read().decode()
        except Exception as e: time.sleep(4)
    return ""
for line in open(sys.argv[1]):
    name=line.strip()
    if not name: continue
    term=urllib.parse.quote(f'"{name}"[Journal] AND ("2026/03/01"[PDAT] : "2026/12/31"[PDAT]) AND journal article[pt] NOT review[pt]')
    r=get(E+f"esearch.fcgi?db=pubmed&term={term}&retmax=300&retmode=json")
    try: j=json.loads(r)["esearchresult"]; ids=j["idlist"]; tot=j["count"]
    except Exception: print(name,"ERR"); continue
    time.sleep(0.4)
    days=[]
    if ids:
        x=get(E+"efetch.fcgi?db=pubmed&retmode=xml&id="+",".join(ids))
        for art in x.split("<PubmedArticle>")[1:]:
            d={}
            for st in ("received","accepted"):
                m=re.search(rf'<PubMedPubDate PubStatus="{st}">\s*<Year>(\d+)</Year>\s*<Month>(\d+)</Month>\s*<Day>(\d+)</Day>',art)
                if m: d[st]=datetime.date(*map(int,m.groups()))
            if len(d)==2 and 0<=(d["accepted"]-d["received"]).days<1500: days.append((d["accepted"]-d["received"]).days)
    if days:
        days.sort()
        print(f"{name:45s} total={tot:>6s} n={len(days):3d} median={statistics.median(days):5.0f} q1={days[len(days)//4]:4d} q3={days[3*len(days)//4]:4d} frac<=72={sum(x<=72 for x in days)/len(days):.2f}")
    else: print(f"{name:45s} total={tot} ids={len(ids)} no dates")
    time.sleep(0.5)

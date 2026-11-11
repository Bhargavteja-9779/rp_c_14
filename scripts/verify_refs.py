"""Verify bibliographic records against Crossref (DOIs) and arXiv; writes manuscript/references_verified.json"""
import json, sys, time, urllib.request, urllib.parse, re
DOIS = """10.1007/s10618-020-00701-z 10.1145/3447548.3467231 10.1007/s10618-022-00844-1 10.1007/s10618-023-00939-3
10.1007/s10618-024-01036-9 10.1007/s10618-024-01022-1 10.1109/JAS.2019.1911747 10.1613/jair.1.12125
10.1109/TNNLS.2013.2292894 10.1109/TNNLS.2022.3152527 10.1109/TSMC.1972.4309137 10.24963/ijcai.2023/450
10.1007/978-3-030-86486-6_29 10.1109/IJCNN.2017.7966039 10.1080/00401706.1979.10489751 10.2307/2346806
10.1007/978-3-642-33460-3_15 10.1609/aaai.v31i1.10894 10.1109/CVPR.2017.240 10.1007/s10618-019-00619-1
10.1073/pnas.1907378117 10.1613/jair.606 10.1007/978-0-387-84858-7 10.1609/aaai.v35i13.17364 10.2307/3001968
10.2202/1544-6115.1000 10.1109/TPAMI.2015.2456899 10.1023/A:1022873112823 10.1007/s10618-016-0483-9
10.1080/00401706.1970.10488634 10.1007/s10994-013-5412-1""".split()
out = {}
for d in DOIS:
    try:
        m = json.load(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(d), headers={"User-Agent": "ref-check/1.0"}), timeout=60))["message"]
        au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", [])[:12])
        yr = (m.get("published-print") or m.get("published") or m.get("issued"))["date-parts"][0][0]
        out[d] = dict(authors=au, year=yr, title=(m.get("title") or [""])[0], venue=(m.get("container-title") or [""])[0],
                      volume=m.get("volume"), issue=m.get("issue"), pages=m.get("page") or m.get("article-number"), n_authors=len(m.get("author", [])))
        print(d, "|", yr, "|", au[:70], "|", out[d]["title"][:70], "|", out[d]["venue"][:40], out[d]["volume"], out[d]["pages"])
    except Exception as e:
        print(d, "FAILED", repr(e)[:80]); out[d] = None
    time.sleep(0.3)
for a in ["2401.15610", "2608.11704", "2410.15954"]:
    try:
        x = urllib.request.urlopen("https://export.arxiv.org/api/query?id_list=" + a, timeout=60).read().decode()
        t = re.findall(r"<title>(.*?)</title>", x, re.S)[-1].strip().replace("\n", " ")
        au = re.findall(r"<name>(.*?)</name>", x); yr = re.findall(r"<published>(\d{4})", x)[0]
        out["arXiv:" + a] = dict(authors="; ".join(au), year=int(yr), title=re.sub(r"\s+", " ", t), venue="arXiv preprint")
        print("arXiv", a, "|", yr, "|", "; ".join(au)[:70], "|", t[:80])
    except Exception as e:
        print("arXiv", a, "FAILED", repr(e)[:80])
    time.sleep(3)
json.dump(out, open("manuscript/references_verified.json", "w"), indent=1)

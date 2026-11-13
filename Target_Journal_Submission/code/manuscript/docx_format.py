"""Post-process the pandoc DOCX into a submission-style layout (Springer Nature generic guidelines:
plain 12-pt serif text, double-spaced-ish, numbered headings typed in the text, tables as editable Word
tables with captions above, figures with captions below, page numbers, continuous line numbers)."""
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

FONT = "Times New Roman"


def _font(run, size=None, bold=None, italic=None):
    run.font.name = FONT
    r = run._element.rPr
    if r is None:
        r = run._element.get_or_add_rPr()
    fonts = r.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts"); r.append(fonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(a), FONT)
    if size: run.font.size = Pt(size)
    if bold is not None: run.font.bold = bold
    if italic is not None: run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def _borders(table):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4" if edge == "insideH" else "8"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)


def _page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), tag); run._r.append(f)
        else:
            i = OxmlElement("w:instrText"); i.set(qn("xml:space"), "preserve"); i.text = text; run._r.append(i)
    _font(run, 10)


def polish(path):
    d = Document(path)
    for s in d.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.left_margin = s.right_margin = Cm(2.5)
        s.top_margin = s.bottom_margin = Cm(2.5)
        ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous")
        s._sectPr.append(ln)
        _page_number(s)
    sizes = {"Title": 16, "Heading 1": 13, "Heading 2": 12, "Heading 3": 12}
    for st in d.styles:
        try:
            st.font.name = FONT
        except Exception:
            pass
    for p in d.paragraphs:
        name = p.style.name
        pf = p.paragraph_format
        if name in sizes:
            for r in p.runs: _font(r, sizes[name], bold=True, italic=False)
            pf.space_before, pf.space_after, pf.keep_with_next = Pt(14 if name != "Title" else 0), Pt(6), True
            if name == "Title": p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            continue
        is_caption = name in ("Image Caption", "Table Caption", "Caption", "Captioned Figure")
        has_pic = bool(p._element.findall(".//" + qn("w:drawing")))
        for r in p.runs: _font(r, 10 if is_caption else 12)
        if is_caption:
            pf.space_before, pf.space_after, pf.line_spacing = Pt(4), Pt(10), 1.0
            pf.keep_with_next = name == "Table Caption"
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif has_pic:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER; pf.keep_with_next = True; pf.line_spacing = 1.0
        elif name in ("Author", "Date"):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        else:
            pf.line_spacing, pf.space_after = 1.5, Pt(6)
            if not p._element.findall(".//" + qn("m:oMathPara")):
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for t in d.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = True
        tblPr = t._tbl.tblPr
        for tag in ("w:tblW", "w:tblLayout"):
            for e in tblPr.findall(qn(tag)): tblPr.remove(e)
        w = OxmlElement("w:tblW"); w.set(qn("w:w"), "5000"); w.set(qn("w:type"), "pct"); tblPr.append(w)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "autofit"); tblPr.append(lay)
        for g in t._tbl.findall(qn("w:tblGrid")):
            for gc in g.findall(qn("w:gridCol")): gc.attrib.pop(qn("w:w"), None)
        for row in t.rows:
            for c in row.cells:
                tcPr = c._tc.get_or_add_tcPr()
                for e in tcPr.findall(qn("w:tcW")): tcPr.remove(e)
                e = OxmlElement("w:tcW"); e.set(qn("w:w"), "0"); e.set(qn("w:type"), "auto"); tcPr.append(e)
        _borders(t)
        wide = len(t.columns) > 8
        for i, row in enumerate(t.rows):
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(0)
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs: _font(r, 7 if wide else 8.5, bold=(i == 0))
    # references: hanging indent, single spacing
    in_refs = False
    for p in d.paragraphs:
        if p.style.name == "Heading 1":
            in_refs = p.text.strip() == "References"
            continue
        if in_refs:
            pf = p.paragraph_format
            pf.left_indent, pf.first_line_indent, pf.line_spacing, pf.space_after = Cm(0.75), Cm(-0.75), 1.0, Pt(4)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs: _font(r, 10.5)
    d.core_properties.title = d.paragraphs[0].text if d.paragraphs else ""
    d.core_properties.author = ""
    d.save(path)


def check(path):
    """Re-open the DOCX and report structural facts used by the formatting audit."""
    d = Document(path)
    pics = sum(len(p._element.findall(".//" + qn("w:drawing"))) for p in d.paragraphs)
    eqs = sum(len(p._element.findall(".//" + qn("m:oMath"))) for p in d.paragraphs)
    heads = [p.text for p in d.paragraphs if p.style.name == "Heading 1"]
    fonts = {r.font.name for p in d.paragraphs for r in p.runs if r.text.strip()}
    return dict(paragraphs=len(d.paragraphs), tables=len(d.tables), figures=pics, equations=eqs, headings=heads, fonts=sorted(f for f in fonts if f))

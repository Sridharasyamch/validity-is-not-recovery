"""Regenerate the IEEE two-column variant FROM paper_vnr/main.tex.

`ieee_main.tex` is a DERIVED file. It once went stale for six days because it was
only ever recompiled, never regenerated, so the IEEE PDF silently lacked three
experiments and several corrections. Never edit ieee_main.tex by hand and never
just run tectonic on it: run this script, then compile.

`src/verify_paper_numbers.py` asserts that ieee_main.tex carries every
\\subsection present in main.tex and is no older than it.
"""
from __future__ import annotations
import os, re, sys

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_vnr")

IEEE_AUTHORS = r"""\author{
\IEEEauthorblockN{Sridhara Syam Chandrabhotla}
\IEEEauthorblockA{Independent Researcher\\Bhimavaram, India\\
\texttt{11sridharasyam@gmail.com}}
\and
\IEEEauthorblockN{KPK Lalitha Vitala}
\IEEEauthorblockA{Dept.\ of Electronics and Communications\\
Vishnu Institute of Technology\\
Bhimavaram, India\\
\texttt{lalitha.v@vishnu.edu.in}}
\and
\IEEEauthorblockN{S.\ Diwakar Bhagavathula}
\IEEEauthorblockA{Dept.\ of Chemistry\\
SRKR Engineering College\\
Bhimavaram, India\\
\texttt{diwakar.b@srkrec.edu.in}}
}
"""


def main():
    s = open(os.path.join(D, "main.tex")).read()

    s = s.replace("""\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}""",
                  """\\documentclass[conference]{IEEEtran}
\\IEEEoverridecommandlockouts""")

    # authblk cannot coexist with IEEEtran's author machinery
    s = s.replace("""\\usepackage{authblk}
\\renewcommand\\Authfont{\\normalsize}
\\renewcommand\\Affilfont{\\small}
""", "")
    # splice by index: the affiliations contain nested braces, and backslashes in
    # the replacement would be read as escapes by re.sub
    _i = s.index("\\author[1]")
    _j = s.index("\\date{}", _i)
    s = s[:_i] + IEEE_AUTHORS + s[_j:]
    s = s.replace("\\date{}", "")

    # every table becomes a full-width float
    s = s.replace("\\begin{table}[t]", "\\begin{table*}[t]").replace(
        "\\end{table}", "\\end{table*}")

    # the two wide figures become full-width floats
    for w in ("fig_candidate_space", "fig_per_category"):
        s = s.replace(
            "\\begin{figure}[t]\n\\centering\n"
            "\\includegraphics[width=0.92\\linewidth]{%s.pdf}" % w,
            "\\begin{figure*}[t]\n\\centering\n"
            "\\includegraphics[width=0.82\\textwidth]{%s.pdf}" % w)
    parts = s.split("\\begin{figure*}[t]")
    out = [parts[0]]
    for p in parts[1:]:
        i = p.index("\\end{figure}")
        out.append(p[:i] + "\\end{figure*}" + p[i + len("\\end{figure}"):])
    s = "\\begin{figure*}[t]".join(out)

    # narrow figures to a single column
    for a, b in [(r"\includegraphics[width=0.78\linewidth]{fig_validity_recovery_plane.pdf}",
                  r"\includegraphics[width=\columnwidth]{fig_validity_recovery_plane.pdf}"),
                 (r"\includegraphics[width=0.8\linewidth]{fig_ranking_ladder.pdf}",
                  r"\includegraphics[width=\columnwidth]{fig_ranking_ladder.pdf}"),
                 (r"\includegraphics[width=0.6\linewidth]{fig_risk_coverage.pdf}",
                  r"\includegraphics[width=\columnwidth]{fig_risk_coverage.pdf}")]:
        s = s.replace(a, b)

    # IEEE abstracts are conventionally a single paragraph; the article version
    # keeps its paragraphing
    s = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
               lambda m: re.sub(r"\n\s*\n+", "\n", m.group(0)), s, flags=re.S)

    # IEEEtran's `description` uses a fixed narrow label width, so \item[...] labels
    # overprint the body text. Convert to enumerate with bold inline labels.
    s = s.replace(r"\begin{description}\itemsep3pt", r"\begin{enumerate}\itemsep3pt")
    s = s.replace(r"\end{description}", r"\end{enumerate}")
    s = re.sub(r"\\item\[\d+\.\s*([^\]]+?)\]", lambda m: r"\item \textbf{" + m.group(1) + "}", s)

    # bare center+tabular blocks are wider than one IEEE column and overprint the
    # neighbouring column; promote them to full-width floats
    def _promote(m):
        body = m.group(1)
        return ("\\begin{table*}[t]\n\\centering\\small\n" + body.strip()
                + "\n\\end{table*}")
    s = re.sub(r"\\begin\{center\}\\small\s*(\\begin\{tabular\}.*?\\end\{tabular\})\s*\\end\{center\}",
               _promote, s, flags=re.S)

    # IEEEtran appends its own colon to \paragraph, giving ".:"
    s = re.sub(r"\\paragraph\{([^}]*?)\.\}", lambda m: r"\paragraph{" + m.group(1) + "}", s)

    s = s.replace(r"\input{table_examples}", r"\input{table_examples_ieee}")
    s = s.replace(r"\bibliographystyle{unsrtnat}",
                  "\\clearpage\n\\bibliographystyle{unsrtnat}")
    open(os.path.join(D, "ieee_main.tex"), "w").write(s)

    t = open(os.path.join(D, "table_examples.tex")).read()
    t = t.replace(r"\begin{table}[t]", r"\begin{table*}[t]").replace(
        r"\end{table}", r"\end{table*}")
    t = t.replace(r"p{2.1cm}p{9.6cm}r", r"p{2.0cm}p{12.4cm}r")
    open(os.path.join(D, "table_examples_ieee.tex"), "w").write(t)

    src = set(re.findall(r"\\subsection\{([^}]*)\}", open(os.path.join(D, "main.tex")).read()))
    dst = set(re.findall(r"\\subsection\{([^}]*)\}", s))
    missing = src - dst
    if missing:
        print("ERROR: sections lost in transformation:", sorted(missing)); sys.exit(1)
    print(f"regenerated ieee_main.tex ({len(dst)} subsections, all of main.tex present)")
    print("authblk removed:", "authblk" not in s, "| IEEEauthorblockN:", s.count("IEEEauthorblockN"))


if __name__ == "__main__":
    main()

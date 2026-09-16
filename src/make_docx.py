"""Build the Word version from paper_vnr/main.tex.

Two transformations pandoc cannot do itself:
  * figures are referenced as .pdf, which Word cannot embed -> point at the
    rasterised .png copies;
  * pandoc does not understand authblk, so \\author[n]{}/\\affil[n]{} silently
    lose every affiliation -> flatten them into a plain author block first.
"""
from __future__ import annotations
import os, re, subprocess, sys

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_vnr")

PLAIN_AUTHORS = r"""\author{
Sridhara Syam Chandrabhotla$^{1}$, KPK Lalitha Vitala$^{2}$, S.\ Diwakar Bhagavathula$^{3}$\\[4pt]
\normalsize $^{1}$Independent Researcher, Bhimavaram, India\\
\normalsize $^{2}$Department of Electronics and Communications, Vishnu Institute of Technology, Bhimavaram, India\\
\normalsize $^{3}$Department of Chemistry, SRKR Engineering College, Bhimavaram, India\\[4pt]
\normalsize All authors contributed equally. Corresponding author: 11sridharasyam@gmail.com\\
\normalsize lalitha.v@vishnu.edu.in \quad diwakar.b@srkrec.edu.in
}
"""


def main():
    s = open(os.path.join(D, "main.tex")).read()
    s = s.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{figures_png/}}")
    s = re.sub(r"\.pdf\}", ".png}", s)
    s = s.replace("\\usepackage{authblk}\n", "")
    s = re.sub(r"\\renewcommand\\(Auth|Affil)font\{[^}]*\}\n", "", s)
    i = s.index(r"\author[1]")
    j = s.index(r"\date{}", i)
    s = s[:i] + PLAIN_AUTHORS + s[j:]
    out = os.path.join(D, "main_docx.tex")
    open(out, "w").write(s)

    r = subprocess.run(["pandoc", out, "-o", os.path.join(D, "main.docx"),
                        "--bibliography", os.path.join(D, "references.bib"),
                        "--citeproc", "--resource-path",
                        f"{D}:{os.path.join(D,'figures_png')}"],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[:800]); sys.exit(1)
    print("built main.docx with affiliations preserved")


if __name__ == "__main__":
    main()

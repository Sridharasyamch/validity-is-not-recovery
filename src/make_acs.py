"""Regenerate the ACS / JCIM variant FROM paper_vnr/main.tex.

`acs_main.tex` is a DERIVED file: never edit it by hand, never just recompile it.
Run this script, then compile. It rewrites the BMC-style manuscript into ACS house
style using the publisher's own `achemso` class: ACS author/affiliation block, a
Table-of-Contents entry, and the back matter JCIM requires (Data and Software
Availability, Author Contributions, Funding Sources, Notes).

`src/verify_paper_numbers.py` asserts that acs_main.tex carries every \\subsection
present in main.tex and is no older than it.
"""
from __future__ import annotations
import os

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_vnr")

ACS_AUTHORS = r"""\author{Sridhara Syam Chandrabhotla}
\altaffiliation{All authors contributed equally.}
\affiliation{Independent Researcher, Bhimavaram, India}
\email{11sridharasyam@gmail.com}
\author{KPK Lalitha Vitala}
\altaffiliation{All authors contributed equally.}
\affiliation{Department of Electronics and Communications, Vishnu Institute of
Technology, Bhimavaram, India}
\author{S.\ Diwakar Bhagavathula}
\altaffiliation{All authors contributed equally.}
\affiliation{Department of Chemistry, SRKR Engineering College, Bhimavaram, India}
\keywords{SMILES; molecular string repair; generative models for molecules;
evaluation methodology; chemical language models; benchmark; neuro-symbolic
methods; selective prediction}
"""

BACK_MATTER = r"""\section*{Data and Software Availability}

The benchmark (6{,}000 corrupted SMILES with ground truth, split into development
and test), the multi-edit and external benchmarks, every per-item system output,
and the code that reproduces every table and figure in this paper are available at
\url{https://github.com/Sridharasyamch/validity-is-not-recovery} under the MIT
licence. The trained generators and the supervised-corrector baseline are released
as assets in the same repository. The enumerated candidate sets and the
245{,}000-pair supervised-training corpus are omitted for size and regenerate
deterministically from the released code. Source molecules come from the public
GuacaMol/ChEMBL corpus. Parsing, sanitisation and all chemistry operations use
RDKit; exact dependency versions are pinned in \texttt{requirements.txt} in the
repository. The baseline corrector weights were trained by us; SmiSelf was run
from its authors' released implementation.

\section*{Author Contributions}

All authors contributed equally to this work. They jointly conceived the study,
designed the benchmark and evaluation protocol, developed and implemented the
method, analysed the results, and wrote and approved the manuscript.

\section*{Funding Sources}

This research received no external funding.

\section*{Notes}

The authors declare no competing financial interest.

\bibliography{references}

\end{document}
"""


def main():
    p = os.path.join(D, "main.tex")
    s = open(p).read()

    # 1. publisher class replaces article + geometry
    s = s.replace("""\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}""",
                  """\\documentclass[journal=jcisd8,manuscript=article]{achemso}""")

    # 2. achemso loads and configures natbib itself; ours would clash
    s = s.replace("""\\usepackage[numbers,sort&compress]{natbib}
\\renewcommand{\\bibfont}{\\small}
\\setlength{\\bibsep}{3pt plus 1pt}
""", "")
    s = s.replace("""\\usepackage{authblk}
\\renewcommand\\Authfont{\\normalsize}
\\renewcommand\\Affilfont{\\small}
""", "")

    # 3. ACS titles are a single line
    s = s.replace(
        "\\title{\\bfseries Validity Is Not Recovery:\\\\Constrained Candidate Selection\\\\for Molecular String Repair}",
        "\\title{Validity Is Not Recovery: Constrained Candidate Selection for\n"
        "Molecular String Repair}")

    # 4. splice the author block by index (nested braces defeat regex)
    i = s.index("\\author[1]")
    j = s.index("\\date{}", i)
    s = s[:i] + ACS_AUTHORS + s[j:]
    s = s.replace("\\date{}\n", "")

    # 5. the ACS Table-of-Contents entry
    s = s.replace("\\begin{document}", """\\begin{document}

\\begin{tocentry}
\\includegraphics[width=3.25in]{toc_graphic.pdf}
\\end{tocentry}""", 1)

    # 6. the visible Keywords line already sits straight after the abstract, which
    #    is where ACS wants it; the \keywords macro stays for ACS tooling. Nothing
    #    to move -- jcisd8 style does not typeset \keywords itself.

    # 7. ACS convention for the opening section
    s = s.replace("\\section{Background}", "\\section{Introduction}")

    # 8. BMC declarations -> ACS back matter
    k = s.index("\\section*{Declarations}")
    s = s[:k] + BACK_MATTER

    out = os.path.join(D, "acs_main.tex")
    open(out, "w").write(s)
    subs = s.count("\\subsection{")
    print(f"  regenerated acs_main.tex ({subs} subsections)")
    for must in ["achemso", "tocentry", "Data and Software Availability",
                 "competing financial interest", "\\keywords{",
                 "\\noindent\\textbf{Keywords:}"]:
        assert must in s, must
    for gone in ["authblk", "\\date{}", "bibliographystyle", "Declarations"]:
        assert gone not in s, f"left over: {gone}"
    print("  preamble, TOC entry and back matter verified")


if __name__ == "__main__":
    main()

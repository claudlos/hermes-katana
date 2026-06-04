Cross-Platform Transferability of Prompt Injection Attacks:
A Taxonomy of Universal Vulnerabilities in Large Language Models
================================================================

arXiv-ready paper (NeurIPS format).

Files
-----
  main.tex           - Paper source
  bibliography.bib   - References (BibTeX)
  neurips.sty        - NeurIPS 2025 style file
  extra_pkgs.tex     - Package imports
  Makefile           - Build system
  archive/cross-platform-transferability-taxonomy-old.pdf
                    - Earlier paper version kept for historical reference

Compile locally
---------------
  make              # produces main.pdf
  make clean        # removes build artifacts

Requires: pdflatex, bibtex (TeX Live or MiKTeX).

Compile via Overleaf
--------------------
1. Go to https://overleaf.com
2. New Project → Upload Project
3. Upload a zip of this `paper/` directory (or the directory itself)
4. Set main.tex as the main document
5. Click Recompile

Submit to arXiv
---------------
1. Compile to produce main.pdf
2. Go to https://arxiv.org/submit
3. Upload: main.tex, bibliography.bib, neurips.sty, extra_pkgs.tex, fig_confusion.tex, fig_roc.tex
4. Select subject: cs.CR (Cryptography and Security) or cs.AI (Artificial Intelligence)
5. arXiv will recompile automatically

Empirical basis
---------------
  2,363 adversarial sessions
  16 model/harness combinations
  5 platforms (local, OpenRouter, Nous, Anthropic, Google)
  17,643 attack corpus (100 stratified shards)

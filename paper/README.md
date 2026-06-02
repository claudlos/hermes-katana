Cross-Platform Transferability of Prompt Injection Attacks
===========================================================

arXiv-ready paper (NeurIPS format).

Files
-----
  main.tex           - Paper source
  bibliography.bib   - References (7 verified citations)
  neurips.sty        - NeurIPS 2025 style file
  extra_pkgs.tex     - Package imports
  Makefile           - Build system

Compile locally
---------------
  make              # produces main.pdf
  make clean        # removes build artifacts

Requires: pdflatex, bibtex (TeX Live or MiKTeX).

Compile via Overleaf
--------------------
1. Go to https://overleaf.com
2. New Project → Upload Project
3. Upload adversarial-llm-paper-arxiv.zip (or the entire directory)
4. Set main.tex as the main document
5. Click Recompile

Submit to arXiv
---------------
1. Compile to produce main.pdf
2. Go to https://arxiv.org/submit
3. Upload: main.tex, bibliography.bib, neurips.sty, extra_pkgs.tex
4. Select subject: cs.CR (Cryptography and Security) or cs.AI (Artificial Intelligence)
5. arXiv will recompile automatically

Empirical basis
---------------
  2,919 adversarial sessions
  18 model/harness combinations
  5 platforms (local, OpenRouter, Nous, Anthropic, Google)
  17,643 attack corpus (100 stratified shards)

# Building the manuscript

## The journal version (what you submit)

`paper4_manuscript.tex` uses **REVTeX 4.2** (`aps,pre,twocolumn`), the APS class
Physical Review requires.

```bash
pdflatex paper4_manuscript
pdflatex paper4_manuscript      # second pass for cross-references
```

The figure PDFs in this directory are the ones the source expects, at the
journal's own column widths (8.6 cm single, 17.2 cm double). **Do not add a
`width=` option to `\includegraphics`**: the fonts are already 7–8 pt at 1:1,
and scaling changes them.

### If the build fails: `File 'revtex4-2.cls' not found`

This is the usual cause of "VS Code produces no PDF". LaTeX Workshop reports it
in the build log rather than on screen, so check the log first — if that line is
there, nothing else is wrong with the document.

REVTeX is not part of a minimal TeX installation. Install it:

| Distribution | Command |
|---|---|
| TeX Live / MacTeX | `tlmgr install revtex` (may need `sudo`) |
| MiKTeX | installs on demand at first build; or MiKTeX Console → Packages → `revtex` |
| Debian/Ubuntu apt | `sudo apt install texlive-publishers` |
| Overleaf | already present, nothing to do |

Verify with `kpsewhich revtex4-2.cls`, which should print a path.

Two further notes for VS Code specifically. LaTeX Workshop's default recipe is
`latexmk`, which is fine here — there is no bibtex step, since the references
are a `thebibliography` block in the source. And if the PDF viewer stays blank
after a successful build, it is usually the recipe writing to an `out/`
directory that the viewer is not looking in.

## The reading copy

`make_preview.py` builds `paper4_manuscript_PREVIEW.pdf` without REVTeX:

```bash
python make_preview.py
```

It is **single column on purpose**. In two columns every float must be hoisted
to the top of a column, and with five figures and three tables competing for
those slots LaTeX defers most of them past the references — which is useless as
a reading copy. In one column each figure sits in the text that discusses it.

Because of that, the preview must not be used to judge the journal version.
Column widths, the title block, table rules (`ruledtabular`), the abstract
format, float placement and the page count all differ, and the figures are
scaled to the text width rather than used at their native size. Delete the
preview from any release.

### One thing that looks wrong in the source and is not

The source places `\begin{abstract}` **before** `\maketitle`. That is correct
for REVTeX, which collects the abstract and prints it after the title block.
Plain `article` would print it where it stands, which is why `make_preview.py`
reorders it — and why an earlier preview appeared to show the abstract above
the title. Do not "fix" the source.

# Paper 1 draft: A Compact Per-Key Network for Video-Only Piano Transcription

Draft for *Signal, Image and Video Processing* (Springer). It follows the journal's rules:
Springer Nature LaTeX template, at most 10 pages in the final two-column format (references
not counted), abstract of 150–250 words (now 240), 4–6 keywords (now 5), numbered references
with DOIs, and a "Statements and Declarations" section.

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper. |
| `figures/fig1_model.pdf`, `fig2_recordings.pdf`, `fig3_register.pdf` | The three figures (the `.png` copies are for the HTML report). |
| `make_figures.py` | Rebuilds the figures from the result files. |
| `data/error_analysis.json` | Recall by register and key colour, and the error classes (per-key network and global variant), used by Fig. 3 and Table 3. |

## Compile

1. Get the Springer Nature LaTeX template: `sn-jnl.cls` and `sn-mathphys-num.bst`
   (Springer's template download, or "Springer Nature LaTeX Template" on Overleaf).
2. Put `main.tex` and the folder `figures/` next to them.
3. Run pdfLaTeX (twice, so that the references and table numbers are resolved).

## Rebuild the figures

```
python paper/make_figures.py --results C:/Users/User/Desktop/PHD/server/results
```

It needs numpy and matplotlib, and reads `E1_vam_full_vs_baseline.csv`,
`A1_vam_strip_vs_baseline.csv` (from the results folder) and `data/error_analysis.json`.

## Still to fill in (marked with [square brackets] in `main.tex`)

- Author names, affiliations and e-mail addresses (and ORCID if wanted).
- Acknowledgements and funding.
- Competing interests: the draft says there are none; please confirm.
- Author contributions.
- The links to the code and to the released files (removed/validation video lists,
  predicted MIDI files, scores per recording).

## Where the numbers come from

Every number in the paper was recomputed from the downloaded result files on 4 Oct 2026
(per-recording CSV files of E1, A1, E2, E3b and E4; p-values with scipy's two-sided Wilcoxon
test). The statements about other work were checked against the papers: V2N
(arXiv 2608.03419), PPAN (IJCAI 2025, arXiv 2411.09037), PianoVAM (ISMIR 2025,
arXiv 2509.08800), Audeo (NeurIPS 2020) and Li et al. (IEEE/ACM TASLP 2024).

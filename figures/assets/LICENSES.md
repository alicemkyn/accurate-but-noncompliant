# Icon provenance and licences

Every icon used in the graphical abstract is recorded here **before** it goes into
a figure. An unlicensed icon in a published figure is a rights problem the authors
own, and this paper's whole argument is that provenance should be recorded rather
than assumed.

Fill one row per file. Do not add an icon to a figure until its row exists.

| File | Depicts | Source | Licence | Attribution required | Retrieved | Modified |
|---|---|---|---|---|---|---|
| _example.svg_ | _kidney_ | _Bioicons (contributor name)_ | _CC0_ | _no_ | _YYYY-MM-DD_ | _recoloured only_ |

## Rules for this directory

1. **CC0 or CC BY only.** Anything with a non-commercial or no-derivatives clause
   is out: the paper will be distributed by the publisher and the figure will be
   redrawn at least once.
2. **No export from a tool whose free tier withholds publication rights.** Drawing
   in such a tool and exporting for a submission breaches its terms even when the
   icon itself is generic.
3. **No generated imagery.** Every number in `make_fig_overview.py` is parsed from
   a result file and the tests enforce it; a generated illustration has no
   comparable provenance and would be the weakest artifact in the paper.
4. **Attribution goes in the figure caption**, not only in this file, for any icon
   whose licence requires it. Keep the wording the licence asks for.
5. **Record modifications.** Recolouring and cropping are modifications; say so.

## Where the icons are allowed to matter

Icons carry no data in this figure. Every quantity stays in text parsed from a
source file, so an icon can be swapped or dropped without changing a single
reported value. If an icon ever becomes load-bearing -- encoding a count, a
direction, or a magnitude -- it needs the same provenance treatment as a number,
which means a reader in the script, not an asset in this folder.

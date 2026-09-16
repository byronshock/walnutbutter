# MNIST by eye: Claude on the first thousand exemplars (September 16, 2026)

Byron, the same night: "How do YOU do on this task on the first thousand exemplars?"

**The task as the network sees it:** the seed-1 stream of the training split,
each image averaged over 2 x 2 blocks to 14 x 14 and a block on iff its mean is
at least half of full (AUTHORITY.md §8; `walnutbutter.mnist`). The first
thousand were rendered a hundred to a sheet, unlabelled, in stream order
(`docs/mnist-glyphs.png` shows the first ten of each class the same way); Claude
read each sheet once and wrote its hundred guesses before any label was looked
at; the labels were then scored exactly. The guesses are at the end of this file.

| reader | right of 1,000 |
|---|---|
| Claude, by eye, one pass | **917** |
| nearest centroid on the same bits (the other 59,000 as reference) | 774 |
| 1-nearest-neighbour, Hamming, on the same bits | 936 |
| 5-nearest-neighbour, Hamming, on the same bits | 929 |

Per class, right of seen: 0 101/102, 1 103/105, 2 87/95, 3 78/90, 4 109/113,
5 92/98, 6 86/94, 7 107/119, 8 73/86, 9 81/98. Per sheet: 94, 84, 86, 96, 97,
92, 88, 92, 91, 97.

The commonest confusions (true -> guessed): 9 -> 4 (6), 8 -> 1 (5), 7 -> 9 (4),
9 -> 3 (4), 8 -> 2 (4), 6 -> 1 (4). Most are digits written thin, which the
half-of-full threshold cuts to slivers and fragments (exemplar 714 is entirely
blank after binarisation). Two runs of errors are transcription, not reading:
exemplars 274-279 and 190-199 were written shifted by one after a glyph was
skipped, about ten of the 83 errors; they are left as scored, since the rule was
guesses before labels.

**What it says for the task:** the 14 x 14 bits carry the digit to about 93-94%
for a memoriser of the rest of the split, and a reader by eye is a point or two
under that at this rendering. The representation is not the ceiling the network
has been hitting; chance is.

## The guesses

```
# Claude's guesses, one row of ten per line, exemplars in stream order; written before any label was looked at
# sheet 0: 0-99
6164016424
4636528803
5443206848
8017082269
6002676806
1334284775
6137719350
9204704611
0974878539
3424585495
# sheet 1: 100-199
5453079709
0487942102
5572618693
1456822414
3646742311
5826100675
9710867746
8765181448
2307525024
1554525739
# sheet 2: 200-299
3646756419
5121045638
0755597767
9273862625
3752668153
8525144573
5625804458
8174316191
1809268355
8175738263
# sheet 3: 300-399
8669513270
2634135707
2936978215
0693009540
9456942699
0834093241
0989284630
9119499158
8643649404
0032814726
# sheet 4: 400-499
9120152902
0831440793
5705075391
7671942102
9755982358
1217465758
7191709099
5467944347
9361250702
3180707249
# sheet 5: 500-599
6278933912
4512914436
7086173115
3203139017
6464997520
1144494337
0315118432
5814907521
1845104524
6914747809
# sheet 6: 600-699
1753784157
6010721205
7742522556
7122744051
3690770096
2042238397
7557872733
8080816761
0193703921
9940783409
# sheet 7: 700-799 (714 is blank: guessed 1)
6714181645
2549173228
5716694146
4792192551
0603615413
2247911531
1538751146
6218113213
3716377911
9534944660
# sheet 8: 800-899
2146729413
3823771016
2816791345
6488857072
2953440348
5245542805
2754137979
2293087870
5600492658
7005449705
# sheet 9: 900-999
4295194457
7618173184
0880019946
1095516886
6140156482
4378090388
1744345010
7029278606
4240772835
8206152031
```

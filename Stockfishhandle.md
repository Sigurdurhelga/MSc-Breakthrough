# Stockfish handling

to get data from stockfish I modified the source code in stockfish (the evaluate.cpp, Eval::evaluate() function) to write data the evaluation for a FEN to a csv file. Stockfish returns the evaluation in some weird format ( digit digit | digit digit | digit digit \n) but that's fine, I write it into a csv and clean it up later.



### cleanup

Firstly stockfish writes to a file called data.csv there and as seen earlier newlines are borked. But I write $ as a delimiter between rows. So clean up data.csv with these commands

```bash
sed ':a;N;$!ba;s/\n//g' data.csv > trimmed.csv
```

This removes all newlines in data.csv (could add -i to command and do it inplace)

secondly now we want to replace $ with newlines using command 

```bash
sed 's/\$/\n/g' trimmed.csv > trimmed2.csv
```

trimmed is still very unclean (the formatting (digit digit | digit digit | digit digit)) this has to be fixed. It is currently fixed using a ipython notebook, see `formatfix.ipynb`

should move to a python script (or look at faster languages because this is annoyingly slow for a 1gb file)

lastly `visualizations.ipynb` has some extra cleanup (throwing away useless columns) this should be merged. 
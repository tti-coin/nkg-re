# nkg-re

# Directory tree
```
nkg-re
│
|-- data .. data for relation extraction with neighborhood KG
│
|-- src .. source code for relation extraction with neighborhood KG
```

install dgl
```
# If you have installed dgl-cuXX package, please uninstall it first.
pip install  dgl -f https://data.dgl.ai/wheels/cu113/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

If you don't have ChemDisGene dataset, please download it from https://github.com/chanzuckerberg/ChemDisGene

train model
```
sh main.sh
```

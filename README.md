# Entity Graph for German

The entity-graph repository provides an implementation for computing, analyzing and visualizing German texts using the entity graph. 

![](images/SCHN_erbschft_1.png)

*Entity graph for first 50 sentences of Die Erbschaft by Arthur Schnitzler.*



## The Entity Graph

The entity graph, first introduced by [Guinaudeau and
Strube (2013)](http://www.aclweb.org/anthology/P13-1010), represents the relations between sentences and entities in a text and thus computes local coherence. We have extended it by adding a syntactic category for possession modifiers and by reducing the weights for entities found in embedded structures. Furthermore, we
extracted a set of graph metrics and show that they are valuable features in text analysis of literary works and author classification.

For more details, please consult the paper by [Suter and Strube (2018)](https://www.oeaw.ac.at/fileadmin/subsites/academiaecorpora/PDF/konvens18_15.pdf). If you would like to use this code for academic work, please cite this paper. 

```
@InProceedings{entgraph2018,  
Title                    = {Extending and Exploiting the Entity Graph for Analysis, Classification and Visualization of German Texts},
  Author                   = {Suter, Julia and Strube, Michael},
  Booktitle                = {Proceedings of the 14th Conference on Natural Language Processing (KONVENS 2018)},
  Year                     = {2018}}
```
## Parsing

This work employs ParZu and CorZu as syntactic parser and coreference resolution systems. For full functionality, install ParZu and CorZu and set the right paths in `settings.py`.

However, the system works without ParZu and CorZu as well. You can pass already parsed texts to the functions as long as they are CoNLL-formatted. 

## Requirements

The system is implemented in Python 3.5 and tested under Ubuntu Linux, but it should work under other environments which have Python installed (no guarantee).

If you want to use the embedded parser functions, please install ParZu and CorZu.

- [ParZu](https://github.com/rsennrich/ParZu)
- [CorZu](https://github.com/dtuggener/CorZu)

For running this code, you will need the following libraries installed:

- numpy
- matplotlib
- scipy
- networkx
 

## Datasets

For evaluating on the sentence reordering tasks, we used the German Tüba/DZ news corpus `Tuebadz-10.0-conll2010_utf8`. 
This corpus provides gold annotations for syntactic parsing and coreference resolution. 
In /Datasets/Tueba_DZ, you can find the single texts with gold annotations, unparsed and parsed with Parzu/CorZu.

- gold_docs
- unparsed_docs 
- parzu_parsed_docs

## How to Run

### Entity Graph for single document

To compute the entity graph and average out degree for a document, you can pass the filename to the  `test_driver` function in `RUN_entity_graph.py`.

Alternatively, you can use the command line. The flag -p indicates that the file was pre-parsed and does not need to be passed to ParZu and CorZu.

```
python RUN_entity_graph.py ./documents/single_docs/KAFKA_Tuersteher.txt
python RUN_entity_graph.py ./documents/single_docs/KAFKA_Tuersteher_parsed.txt -p 
```
Per default, the test driver function will create the entity graph with networkx and save it in the /entity_graph directory. You can disable this by setting `draw_single_graph` to False. 

### Evaluation

The evaluation script will evaluate the entity graph on the sentence reordering paper as described in the paper. Simply run the script with python. 

In the `settings.py`, you can select the one-mode projection:
Set $version$ to:

- *pu* for P unweighted
- *pw* for P weighted
- *pacc* or *full* for P accumulated

You can also activate or deactivate the possession modifiers and weight reduction extensions. 


Original version
```
possession_modifiers = False
reduce_weights = False
```

Adjusted version
```
possession_modifiers = True
reduce_weights = True
```

Without possession modifiers
```
possession_modifiers = False
reduce_weights = True
```

Without weight reduction
```
possession_modifiers = True
reduce_weights = False
```

Of course, you can change many other settings, for instance the weights for the different syntactic role categories. 


### Graph metrics and author/genre classification

The code for these topics will be added shortly. 
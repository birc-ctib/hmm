# A hidden Markov model gene-finder

In the exercise below, you will implement and experiment with an example of how to apply a HMM for identifying coding regions(genes) in genetic material. We consider only procaryotes, which have a particular simple gene format. A gene is a sequence of triplets, codons, that encode proteins. We saw this in the first project. Now, we assume that we have a genomic sequence, and our goal is to recognise which part of the genome encodes genes, and which do not.

Genes can be found on both strands of a genome, so as we scan along the genome sequence, we might encounter them in the direction we are scanning, or we might encounter them in the reverse order, seeing them backwards, so to speak. In either case, we recon that gene coding sequences likely have a different nucleotide composition, and we will exploit this in our model.

We build a hidden Markov model, where the hidden states are:

- Non-coding: we are not in a coding sequence
- Coding: We are inside a gene that is coded in the direction we are reading
- Reverse coding: We are inside a gene that is encoded on the other strand than then one we are scanning

We will use two different models. A 3-state HMM that encodes only the three states we just listed, and a 7-state HMM that adds a little more structure. It models that genes come as codons, so when we are inside a gene, we should always a number of nucleotides divisible by three, and if the nucleotide composition is different for different codon positions, it can model that as well.

We can draw the two models like this:

![Gene-finder models](figures/gene-finder-models.png)


Both have coding states (C), non-coding states (N), and reverse-coding states (R), but the second has three of each of C and R. The numbers in square brackets in the states are just that, numbers. Specifically numbers from zero up to the number of states minus one. Representing states like that is convinient if we are to use states to index into vectors and matrices.

There is another purpose to this project, beyond learning how to implement and apply hidden Markov models. We will do the project in a so-called [Jupyter Notebook](https://jupyter.org). Jupyter is one of many ways of combining documentation and code, a technique known as literate programming although it is rarely used in programming but quite frequently in data science. The text you are reading now is, in fact, written in a Jupyter Notebook that contains both the project description and the template code you need to get started. The file is named `hmm.ipynb`, and you should edit that file to fill out the missing details.

**WARNING:** DO NOT EDIT `README.md`. When you commit to GitHub, `README.md` will be overwritten by a file extracted from `hmm.ipynb`. The only file you should edit in this project is `hmm.ipynb` (or any additional files you create for testing the code you write in `hmm.ipynb`).

You can edit Jupyter Notebooks in several different ways. If you follow the link above to [jupyter.org](https://jupyter.org) you can get a browser interface. If you are using VSCode as your editor, you can install an extension and edit Jupyter files natively. (That is what I am doing). Your first exercise is to figure out how to edit `hmm.ipynb`. When you have managed that, proceed to the next section.

Whenever you commit the notebook to GitHub it is evaluated from scratch, and the result becomes the new README.md. If you are evaluating cells out of order, your state might be different from one where you evaluate the notebook from the beginning, so be careful to check that the clean evaluation is correct.

## Training data

For this project, we have two bacterial genomes that someone has painstakingly annotated with genes (and I have then fracked up the good work to make the data look the way our models assume it will rather than the mess that God created).

The two genomes, found in `data/genome1.fa` and `data/genome2.fa`, are stored in [FASTA files](https://en.wikipedia.org/wiki/FASTA_format), and I have provided a function you can used to load them. (It is not entirely general, but it suffices for parsing the FASTA files you need for this project).

The function is in a module in `src` and we can import it like this:


```python
# load modules from src directory
import sys
sys.path.insert(0, 'src')
```


```python
# get the function for reading a fasta file
from fasta import read_fasta_file
```

With the function in hand, we can try to load one of the genomes and see what is in the file:


```python
genome1 = read_fasta_file('data/genome1.fa')
print(genome1.keys())
```

    dict_keys(['genome', 'annotation'])


It looks like you get a dictionary with two keys, `genome` and `annotation`, and I will bet you good money that the former is the genomic sequence and the second the annotation. Let's have a look at them (but don't print all of them, as they are quite long):


```python
print(len(genome1['genome']), len(genome1['annotation']))
print(genome1['genome'][:60])
print(genome1['annotation'][:60])
```

    1852441 1852441
    TTGTTGATATTCTGTTTTTTCTTTTTTAGTTTTCCACATGAAAAATAGTTGAAAACAATA
    NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN


The genomic sequence is a sequence over the letters:


```python
print(set(genome1['genome']))
```

    {'C', 'A', 'G', 'T'}


while the annotation is a sequence over the letters


```python
print(set(genome1['annotation']))
```

    {'C', 'R', 'N'}


that should be interpreted as non-coding, reverse-coding, and coding.

If we are to analyse such a genome using a hidden Markov model, we must be able to use both the observable letters (the genomic sequence) and the hidden states (the annotation) as indices into matrices or vectors.

For the three-states model, the $\pi$ vector should have length three, and we should be able to index into it with hidden states, $\pi[z]$; we should be able to index into the transition matrix $T$, a $3\times 3$ matrix, with two hidden states $T[s,t]$, and we should be able to index into the emission matrix, a $3\times 4$ matrix, with a hidden state $z$ and an observed nucleotide, $x$, $E[z,x]$.

We could use dictionaries for this indexing, but it is much more convinient to represent vectors as vectors and matrices as matrices (we will see how below), and that requires that we use integers as indices. We need to map the two strings into lists of integers, in some way such that for the genomic sequence the integers are 0, 1, 2, or 3, and such that the annotation maps to integers 0, 1, or 3.

I'll show you have to map the genomic sequence, and then you should write a function for mapping the annotation.


```python
def observed_states(x: str) -> list[int]:
    """
    Maps a DNA sequence to HMM emissions.
    
    >>> observed_states('ACAGTTC')
    [0, 1, 0, 2, 3, 3, 1]
    """
    map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [map[a] for a in x]

def rev_observed_states(obs: list[int]) -> str:
    """
    Reverses the observable states mapping.
    
    In this notebook, we only use this for testing, but you can use it as
    inspiration to write similar functions for the hidden states where you
    do want to be able to reverse when you do decoding.
        
    >>> rev_observed_states([0, 1, 0, 2, 3, 3, 1])
    'ACAGTTC'
    """
    return ''.join('ACGT'[x] for x in obs)

```


```python
x = genome1['genome'][:10] # A shorter string to play with
y = observed_states(x)
print('mapped genome:', y)
print(x, rev_observed_states(y))
assert x == rev_observed_states(y)
```

    mapped genome: [3, 3, 2, 3, 3, 2, 0, 3, 0, 3]
    TTGTTGATAT TTGTTGATAT



```python
# FIXME: You need to implement the corresponding function for annotations
# In the figure above, the states map as C -> 0, N -> 1 and R -> 2 but
# you do not need to use that; but you do need to be consistent everywhere
def hidden_states(x: str) -> list[int]:
    """
    Map a genome annotation to hidden states (index 0, 1, or 2).
    
    >>> hidden_states('NNCCCCCCNNRRRRRRN')
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1]
    """
    map = {'C': 0, 'N': 1, 'R': 2}
    return [map[a] for a in x]

def rev_hidden_states(hid: list[int]) -> str:
    """
    Reverse the map of hidden states.
    
    This function should also be useful if you wish to convert a decoding
    to the annotation format at some point in the future.
    
    >>> rev_hidden_states([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1])
    'NNCCCCCCNNRRRRRRN'
    """
    return ''.join("CNR"[h] for h in hid)

```


```python
x = genome1['annotation'][220:250] # A shorter string to play with
y = hidden_states(x)
print('mapped annotation:', y)
print(x, rev_hidden_states(y))
assert x == rev_hidden_states(y)
```

    mapped annotation: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    NNNNNNNNNNNCCCCCCCCCCCCCCCCCCC NNNNNNNNNNNCCCCCCCCCCCCCCCCCCC


There is a problem for the seven-state HMM, though. Our annotations have three states, `C`, `N`, and `R`, but the `C` and `R` annotations map to three separate states each in the seven-state model!

It's not that it is a complete mystery which hidden states the different `C` or `R` annotations should map to. The coding regions come in triplets, so if we split a stretch of `C` positions into triplets

```
    ...NNCCCCCCCCCCCCCCCNN...
=>  ...NN[CCC][CCC][CCC][CCC][CCC]NN...
```

they should obviously be mapped to repeats of `[0,1,2]` (according to the figure above), so this annotation should be interpreted as

```
    ...NN[CCC][CCC][CCC][CCC][CCC]NN...
=>  ...33[012][012][012][012][012]33...
```

Likewise for stretches of `R` annotations.

There are many fine ways to achieve this. A simple one is to look at the annotation and hidden state in the previous position and set the hidden state at the current position based on that.

```python
 if ann[i] == 'C' and ann[i-1] != 'C':
     hid[i] = 0
 if ann[i] == 'C' and ann[i-1] == 'C':
     hid[i] = (hid[i-1] + 1) % 3
```

I don't really care how you do it, but I want it done. Write me a function that extracts the hidden states from a seven-state model.


```python
# FIXME: You need to implement the corresponding function for annotations
# In the figure above, the states map as C -> 0/1/2, N -> 3 and R -> 4/5/6 but
# you do not need to use that; but you do need to be consistent everywhere
def hidden_states7(x: str) -> list[int]:
    """
    Map a genome annotation to hidden states.
    
    >>> hidden_states7('NNCCCCCCNNRRRRRRN')
    [3, 3, 0, 1, 2, 0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6, 3]
    """
    ann = [-1] * len(x)
    for i, a in enumerate(x):
        match a:
            case 'N': ann[i] = 3
            case 'C' if x[i - 1] != 'C':
                ann[i] = 0
            case 'C' if x[i - 1] == 'C':
                ann[i] = (ann[i - 1] + 1) % 3
            case 'R' if x[i - 1] != 'R':
                ann[i] = 4
            case 'R' if x[i - 1] == 'R':
                ann[i] = (ann[i - 1] -4 + 1) % 3 + 4
    return ann

def rev_hidden_states7(hid: list[int]) -> str:
    """
    Reverse the map of hidden states.
    
    This function should also be useful if you wish to convert a decoding
    to the annotation format at some point in the future.
    
    >>> rev_hidden_states7([3, 3, 0, 1, 2, 0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6, 3])
    'NNCCCCCCNNRRRRRRN'
    """
    return ''.join("CCCNRRR"[h] for h in hid)

```


```python
x = genome1['annotation'][45110:45210]  # A shorter string to play with
y = hidden_states7(x)
print('mapped annotation:', y)
print(x, rev_hidden_states7(y))
assert x == rev_hidden_states7(y)

```

    mapped annotation: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4]
    NNNNNNNNNNNNRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR NNNNNNNNNNNNRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR


## Computing likelihoods

Now that we have transformed our input sequences into integer lists, we should be able to use these with a hidden Markov model. Let's make some (arbitrary) HMM parameters--`pi`, `T` and `E`--to see how.

We will use the module `numpy` for this. It is a powerful library for linear algebra, but don't worry, we will just use it to make one- and two-dimensional tables that we index into efficiently. 


```python
import numpy as np
pi = np.array([0, 1, 0]) # always start in N (== 1)
T = np.array([
    [0.8, 0.2, 0.0],  # Transitions out of C
    [0.1, 0.8, 0.1],  # Transitions out of N
    [0.0, 0.2, 0.8],  # Transitions out of R
])
E = np.array([
    [0.3, 0.2, 0.1, 0.4],  # Emissions from C
    [0.5, 0.1, 0.2, 0.2],  # Emissions from N
    [0.2, 0.2, 0.3, 0.3],  # Emissions from R
])
```

The parameters here are not chosen to fit the model (which should be obvious from how regular the numbers look), but they are valid parameters in the sense that the initial transitions sum to one, that the out transitions sum to one for each state, and that the emissions sum to one for each state as well.


```python
from numpy.testing import assert_almost_equal
assert_almost_equal(sum(pi), 1) # use "almost equal" on floats; never ==
for s in [0, 1, 2]:
    assert_almost_equal(sum(T[s,:]), 1)
for s in [0, 1, 2]:
    assert_almost_equal(sum(E[s, :]), 1)

```

As represented here, we have the three parameters floating around independently. There isn't anything wrong with that as such, but there are occations where you want to make sure that the parameters you have fitted for a model do not get mixed up with parameters from elsewhere, and it is easier to keep one object under control than three. The code below lets you wrap up the three parameters we use as a single object that we can pass around functions.

You don't need to know how it works (although it isn't that complicated), as long as you know that you can call the function `hmm_params(pi,T,E)` to wrap the three parameters, and from the wrapped object, `theta`, you can get the parameters back, including a number `K` that is the number of hidden states.

```python
K, pi, T, E = theta
```

The number `K` is useful in some of the algorithms, and although you can always get it from the other three, it is more convinient to have it directly.


```python
from numpy.typing import ArrayLike
from typing import NamedTuple

HMMParam = NamedTuple('HMMParam', [
    ('K', int),          # Number of hidden states
    ('pi', ArrayLike),
    ('T', ArrayLike),
    ('E', ArrayLike)
])

def hmm_params(pi: ArrayLike, T: ArrayLike, E: ArrayLike) -> HMMParam:
    """Wraps HMM parameters in a tuple so we can pass them to functions
    as a unit."""
    # Consistency check.
    assert len(pi.shape) == 1  # must be one dimensional
    assert len(T.shape) == 2   # must be two dimensional
    assert len(E.shape) == 2   # must be two dimensional

    # Get the number of states from pi and check that
    # it matches with the expected dimensions in T and E.
    K = len(pi)
    assert T.shape == (K, K)
    assert E.shape[0] == K
    
    # Consistency check done, we accept and wrap the parameters
    return HMMParam(K, pi, T, E)

theta = hmm_params(pi, T, E) # combining parameters from above


```

The probability that we go from state $s$ to state $t$ is $T[s,t]$, so with our mapped sequences, let's call them `obs` for observed and `hid` for hidden, the probability of the transition at position `i` should be `T[hid[i],hid[i+1]]`. The probabiity of emitting what we have at position is `E[hid[i],obs[i]]`, and the probability of starting in the first state it `pi[hid[0]]`.

Use these observations to implement a function that computes the likelihood of a genomic sequence and an annotation.


```python
def lik(obs: list[int], hid: list[int], theta: HMMParam) -> float:
    """
    Compute the likelihood of the data (obs,hid) given the parameters, theta.
    
    >>> pi = np.array([0, 1, 0]) # always start in N (== 1)
    >>> T = np.array([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]])
    >>> E = np.array([[0.3, 0.2, 0.1, 0.4], [0.5, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3]])
    >>> theta = hmm_params(pi, T, E)
    >>> lik(observed_states('A'), hidden_states('N'), theta)
    0.55
    """
    _, pi, T, E = theta # get the parameters (we don't need K in this function)
    # FIXME: compute the likelihood
    p = pi[hid[0]]
    for i, s in enumerate(hid[1:]):
        p *= T[hid[i], s]
    for i, _ in enumerate(hid):
        p *= E[hid[i], obs[i]]
    return p

```


```python
obs = observed_states('A')#ACAAGTCA')
hid = hidden_states('N')#NCCCNRR')
print(lik(obs, hid, theta))
print(theta)
```

    0.5
    HMMParam(K=3, pi=array([0, 1, 0]), T=array([[0.8, 0.2, 0. ],
           [0.1, 0.8, 0.1],
           [0. , 0.2, 0.8]]), E=array([[0.3, 0.2, 0.1, 0.4],
           [0.5, 0.1, 0.2, 0.2],
           [0.2, 0.2, 0.3, 0.3]]))


## Testing

Using doc-test to test the functions in the notebook.


```python
import doctest
doctest.testmod(verbose=True)
```

    Trying:
        hidden_states('NNCCCCCCNNRRRRRRN')
    Expecting:
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1]
    ok
    Trying:
        hidden_states7('NNCCCCCCNNRRRRRRN')
    Expecting:
        [3, 3, 0, 1, 2, 0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6, 3]
    ok
    Trying:
        pi = np.array([0, 1, 0]) # always start in N (== 1)
    Expecting nothing
    ok
    Trying:
        T = np.array([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]])
    Expecting nothing
    ok
    Trying:
        E = np.array([[0.3, 0.2, 0.1, 0.4], [0.5, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3]])
    Expecting nothing
    ok
    Trying:
        theta = hmm_params(pi, T, E)
    Expecting nothing
    ok
    Trying:
        lik(observed_states('A'), hidden_states('N'), theta)
    Expecting:
        0.55
    **********************************************************************
    File "__main__", line 9, in __main__.lik
    Failed example:
        lik(observed_states('A'), hidden_states('N'), theta)
    Expected:
        0.55
    Got:
        0.5
    Trying:
        observed_states('ACAGTTC')
    Expecting:
        [0, 1, 0, 2, 3, 3, 1]
    ok
    Trying:
        rev_hidden_states([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1])
    Expecting:
        'NNCCCCCCNNRRRRRRN'
    ok
    Trying:
        rev_hidden_states7([3, 3, 0, 1, 2, 0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6, 3])
    Expecting:
        'NNCCCCCCNNRRRRRRN'
    ok
    Trying:
        rev_observed_states([0, 1, 0, 2, 3, 3, 1])
    Expecting:
        'ACAGTTC'
    ok
    3 items had no tests:
        __main__
        __main__.HMMParam
        __main__.hmm_params
    6 items passed all tests:
       1 tests in __main__.hidden_states
       1 tests in __main__.hidden_states7
       1 tests in __main__.observed_states
       1 tests in __main__.rev_hidden_states
       1 tests in __main__.rev_hidden_states7
       1 tests in __main__.rev_observed_states
    **********************************************************************
    1 items had failures:
       1 of   5 in __main__.lik
    11 tests in 10 items.
    10 passed and 1 failed.
    ***Test Failed*** 1 failures.





    TestResults(failed=1, attempted=11)



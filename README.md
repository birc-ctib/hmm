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



# A hidden Markov model gene-finder

In the exercise below, you will implement and experiment with an example of how to apply a HMM for identifying coding regions(genes) in genetic material. We consider only procaryotes, which have a particular simple gene format. A gene is a sequence of triplets, codons, that encode proteins. We saw this in the first project. Now, we assume that we have a genomic sequence, and our goal is to recognise which part of the genome encodes genes, and which do not.

Genes can be found on both strands of a genome, so as we scan along the genome sequence, we might encounter them in the direction we are scanning, or we might encounter them in the reverse order, seeing them backwards, so to speak. In either case, we recon that gene coding sequences likely have a different nucleotide composition, and we will exploit this in our model.

We build a hidden Markov model, where the hidden states are:

- Non-coding: we are not in a coding sequence
- Coding: We are inside a gene that is coded in the direction we are reading
- Reverse coding: We are inside a gene that is encoded on the other strand than then one we are scanning

We will use two different models. A 3-state HMM that encodes only the three states we just listed, and a 7-state HMM that adds a little more structure. It models that genes come as codons, so when we are inside a gene, we should always a number of nucleotides divisible by three, and if the nucleotide composition is different for different codon positions, it can model that as well.


_l0_-sampler data structure.

Contains sketched info about integer-valued vector _a_ of length _n_
allowing to make linear updates to _a_ and get _l0_-samples from it.
Successfully returns non-zero item _a_i_ with probability at least _1-n^(-c)_,
and conditioned on successful recovery, outputs item _i_ with probability
_1/N +- n^(-c)_, where _N_ is _l0_ norm of _a_, and  _c_ is some positive constant.

Requires O(log(n)^4) space.


For more detailed info check out

Graham Cormode and Donatella Firmani.
"On unifying the space of l0-sampling algorithms."
https://pdfs.semanticscholar.org/b0f3/336c82b8a9d9a70d7cf187eea3f6dbfd1cdf.pdf
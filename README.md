#### what is this?
Another try on a generic AI

#### components

* library lib/es0.nim learning algorithm: evolutionary strategies

* NN training scripts
* Nim program to generate training data for "arithmetic NN"

#### Implemented ML learning algorithms

**Modern Hopfield Networks**

[code](https://github.com/PtrMan/23I/blob/main/HopfieldNn0.py)

**LM ; SynthRASM**

[code](https://github.com/PtrMan/23I/blob/main/lmSynthRasm.py)

goal was to have a LM which is fast to train and archives a high performance. The actual code was generated with Gemini. Training leads to a model which seems to not work / overfit? .

**Fast Weight Programmer**

[code](https://github.com/PtrMan/23I/blob/1b9cf95a9a275ce1b10545648af9018c99fef6d1/FastWeightProgrammerA.py)

Result: FAIL. Gets stuck with high training loss=5.1  on 25kb of training data. Reason is probably that the code to store the gradients in the trace is broken. Gave up on it because the implementation is way to complicated to get working because of the interaction of the trace and slowness of pure SGD.

#### TODO
* add reasoner

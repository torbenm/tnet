# TNET

Some easy neural network implementations in c.

## Dependencies

- Cmake (install with `brew install cmake`)
- Probably just works on MacOS

## Building

```
cmake .
make
```

## Running

After building, run the following for an MLP XOR example:

```
.bin/tnet seq xor
```

Other Options are `perceptron or`, `perceptron and`, `perceptron xor`.
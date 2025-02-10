# TNET

This repository contains some playful neural network implementations in C. They are designed to be somewhat modularized and able to generalize, but this is definitely not clean code.
The mantra of this implementation is "everything from scratch". Apart from some basic stdlib-functions, there are no third-party libraries.

Features currently included:
- `tensors` as data representation. A number of operations on tensors are implemented, such as tensor dot multiplication, element wise operations, etc.; not all of them in a fully generalized way, though. Most of the implementations should however remark where they do not generalize.
- mean squared error and absolute sum as `loss functions`
- A number of `activation functions` (such as heaviside, softmax, sigmoid, tanh, relu, ...) + their derivatives.
- `standard gradient descent` with momentum as the only optimization algorithm
- The `perceptron` as a basic demo
- A generic `sequential model` that allows you to add as many sequential layers as you want
- Two kinds of layers, `input layer` and `fully-connected` / `dense layer`
- Some examples, mostly around the `XOR` operator as this is a nice demnostration of the capabilities of the sequential model vs the perceptron
- A very basic (mark-and-sweep'ish) garbage collector. I'll explain a bit more below.

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
./bin/tnet seq xor
```

Other Options are `perceptron or`, `perceptron and`, `perceptron xor`.

## 'Mark-and-Sweep'-ish Garbage Collector

In early implementations, it was easy to lose track of used and unused objects, such as tensors. 
The way tensors are implemented right now is that the right hand operator is updated in-place whereever possible, 
and as such you need to copy the whole tensor before the operation if you do not want it to alter. 
As such, you might have a lot of orphan tensors, which are hard to detect and to clean up. 
This lead to many memory leaks in early implementations.

There exist generalized garbage collectors in C (such as the [mark-and-sweep algorithm](https://maplant.com/2020-04-25-Writing-a-Simple-Garbage-Collector-in-C.html)),
but at the time of implementation I wanted to have something quick and performant, albeit perhaps not developer friendly.

As such, the collector works as follows: Whenever you allocate a new object (such as a tensor), the pointer to this newly generated object is stored in an array.
This array has a fixed length at the moment, to avoid & detect memory leaks, and as such you might e.g. at most create 4096 objects.
For larger models this will have to be adjusted.
The algorithm then works as the mark-and-sweep algorithm, with the difference that you _tell_ it what pointers to mark.
It does not check the whole stack & heap for potential pointers, it only relies on what you tell it.
The sweep algorithm then removes all unmarked objects. 
This might e.g. be applied after each training iteration, where you usually just want to preserve the training examples, current weights, and perhaps some information about the previous run. 

This is not great, it requires deep knowledge of the objects you want to preserve, it probably also does not scale very well (maximum limit of objects at the moment!);
but it is also really fast. Perhaps I'll move to something more developer friendly soon.


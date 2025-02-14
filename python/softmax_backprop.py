import numpy as np


def original(forward_out, backward_in):
    # Ensure that everything is in the right shape
    softmax = np.reshape(forward_out, (1, -1))
    grad = np.reshape(backward_in, (1, -1))

    d_softmax = (
        softmax * np.identity(softmax.size)
        - softmax.transpose() @ softmax)
    backward_out = (grad @ d_softmax).ravel()
    return backward_out


def softmax_backward(activations, previousSmallDelta):
        activations = np.reshape(activations, (1, -1))
        previousSmallDelta =  np.reshape(previousSmallDelta, (1, -1))
        print("PREDICTIONS")
        print(activations)
        print("GRAD: ")
        print(previousSmallDelta);

        s_t = activations.transpose()
        print("s_t: ")
        print(s_t)

        s_diag = activations * np.identity(activations.size)
        print("s_diag: ")
        print(s_diag)

        s_mul = s_t @ activations
        print("s_mul:")
        print(s_mul)

        s_diag = s_diag - s_mul
        print("elem_sub:")
        print(s_diag)



        # # Ensure that everything is in the right shape
        # # softmax = np.reshape(activations, (1, -1))
        # # grad = np.reshape(previousSmallDelta, (1, -1))
        # # print(f"softmax={softmax}; grad={grad}")
        
        # d_softmax = (softmax * np.identity(softmax.size) - softmax.transpose() @ softmax)
        # print(f"d_softmax=\n{d_softmax}")
        newSmallDelta = (previousSmallDelta @ s_diag).ravel()
        print(f"Output\n{newSmallDelta}")

activations = np.array([0.4717, 0.5283])
truth = np.array([1, 0])

softmax_backward(activations, activations-truth)
print(f"Original:\n{original(activations, activations-truth)}")
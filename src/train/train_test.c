#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "core.h"
#include "funcs.h"
#include "models.h"
#include "train.h"

#define TEST_FAILURE 0
#define TEST_SUCCESS 1
#define E 0.000001

// Some Gradient Checking happening here
int test_dense_layer_gradients()
{
    param_t inputs[2] = {1.0, 0.0};
    param_t truths[2] = {1.0, 0.0};

    tensor *t_inputs = t_from_1dim_array(2, inputs);
    tensor *t_truths = t_from_1dim_array(2, truths);

    // example dense layer model
    struct seqmodel *s = seqmodel_init();
    seqmodel_push(s, input_layer_init(2));
    seqmodel_push(s, dense_layer_init(2, 2, av_tanh));
    seqmodel_push(s, dense_layer_init(2, 2, av_identity));
    seqmodel_push(s, softmax_layer_init());

    // Normal Pass
    tensor **predictions = malloc(sizeof(tensor *));
    struct forwardstate *forwardstates = opt_forwardpropagate(s, t_inputs, predictions);

    struct backwardstate **backwardstates = opt_backwardpropagate(s, predictions[0], t_truths, forwardstates);
    param_t normal_loss = seqmodel_calculate_loss(s, 1, &t_inputs, &t_truths, loss_mse);

    struct denselayer_props *dp;
    param_t numerator = 0;
    param_t denominator = 0;
    for (int l = 1; l <= 2; l++)
    {
        dp = (struct denselayer_props *)s->layers[l]->layerProps;
        for (int i = 0; i < dp->weights->_v_size; i++)
        {
            param_t original = dp->weights->v[i];
            // plus
            dp->weights->v[i] = original + E;
            param_t lossplus = seqmodel_calculate_loss(s, 1, &t_inputs, &t_truths, loss_mse);

            // minus
            dp->weights->v[i] = original - E; // remove twice to account for above
            param_t lossminus = seqmodel_calculate_loss(s, 1, &t_inputs, &t_truths, loss_mse);
            dp->weights->v[i] = original; // undo

            param_t ag = (lossplus - lossminus) / (2 * E);
            param_t cg = backwardstates[l]->weightGradients->v[i];
            printf("L=%i; W[%i]: ag=%f; cg=%f\n", l, i, ag, backwardstates[l]->weightGradients->v[i]);
            param_t num = cg - ag;

            numerator += pow(pow(num, 2.0), 0.5);
            denominator += pow(pow(ag, 2.0), 0.5) + pow(pow(cg, 2.0), 0.5);
        }

        for (int i = 0; i < dp->bias->_v_size; i++)
        {
            param_t original = dp->bias->v[i];
            // plus
            dp->bias->v[i] = original + E;
            param_t lossplus = seqmodel_calculate_loss(s, 1, &t_inputs, &t_truths, loss_mse);

            // minus
            dp->bias->v[i] = original - E; // remove twice to account for above
            param_t lossminus = seqmodel_calculate_loss(s, 1, &t_inputs, &t_truths, loss_mse);
            dp->bias->v[i] = original; // undo

            param_t ag = (lossplus - lossminus) / (2 * E);
            param_t cg = backwardstates[l]->biasGradients->v[i];
            printf("L=%i; b[%i]: ag=%f; cg=%f\n", l, i, ag, backwardstates[l]->biasGradients->v[i]);
            param_t num = cg - ag;

            numerator += pow(pow(num, 2.0), 0.5);
            denominator += pow(pow(ag, 2.0), 0.5) + pow(pow(cg, 2.0), 0.5);
        }
    }
    param_t diff = numerator / denominator;
    printf("Total difference: %f\n", diff);

    if (diff < 1e-7)
        return TEST_SUCCESS;
    else
        return TEST_FAILURE;
}

// Utils
void *__test_thread_train(void *testFn)
{
    int *ret = malloc(sizeof(int));
    ret[0] = ((int (*)(void))testFn)();
    pthread_exit(ret);
}

void __run_test_train(const char *testName, int (*testFn)(void))
{
    pthread_t testThread;
    void *ret;
    printf("Running test case '%s' ..... ", testName);

    if (pthread_create(&testThread, NULL, __test_thread_train, testFn) != 0)
    {
        error("failed to start thread.\n");
    }

    if (pthread_join(testThread, &ret) != 0)
    {
        error("failed to join thread.\n");
    }
    if (ret != NULL && ((int *)ret)[0] == TEST_SUCCESS)
        printf(" ✅");
    else
        printf(" ❌");
    printf("\n");
    free(ret);
}

void train_test()
{
    __run_test_train("test_dense_layer_gradients", test_dense_layer_gradients);
}
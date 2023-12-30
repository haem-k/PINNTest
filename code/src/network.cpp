#include "network.h"

FCImpl::FCImpl(int in_channel) : linear1(register_module("linear1", Linear(LinearOptions(in_channel, 256)))),
                                 linear2(register_module("linear2", Linear(LinearOptions(256, in_channel))))
{
}

Tensor FCImpl::forward(Tensor x)
{
    x = torch::relu(linear1(x));
    x = linear2(x);
    return x;
}


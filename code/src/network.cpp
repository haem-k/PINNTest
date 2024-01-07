#include "network.h"
using namespace torch::indexing;

FCImpl::FCImpl(int in_channel, int out_channel) : linear1(register_module("linear1", Linear(LinearOptions(in_channel, 128)))),
                                                  linear2(register_module("linear2", Linear(LinearOptions(128, 128)))),
                                                  linear3(register_module("linear3", Linear(LinearOptions(128, 128)))),
                                                  linear4(register_module("linear4", Linear(LinearOptions(128, out_channel))))
{
}

Tensor FCImpl::forward(Tensor x)
{
    x = torch::tanh(linear1(x));
    x = torch::tanh(linear2(x));
    x = torch::tanh(linear3(x));
    x = linear4(x);
    return x;
}

GradImpl::GradImpl(FC fcnet) : fcnet(register_module("fcnet", FC(fcnet)))
{
}

std::tuple<Tensor, Tensor, Tensor> GradImpl::forward(Tensor x)
{
    // Forward through fcnet
    Tensor r = fcnet(x);

    // Compute second derivative of r
    Tensor drdt = torch::autograd::grad({r}, {x}, {torch::ones_like(r)}, true, true)[0].set_requires_grad(true);
    Tensor d2rdt2 = torch::autograd::grad({drdt}, {x}, {torch::ones_like(drdt)}, true, true)[0];

    drdt = drdt.index({Slice(), Slice(0, 1)});
    d2rdt2 = d2rdt2.index({Slice(), Slice(0, 1)});

    return {r, drdt, d2rdt2};
}
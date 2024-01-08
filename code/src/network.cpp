#include "network.h"

FCImpl::FCImpl(int in_channel, int out_channel) : linear1(register_module("linear1", Linear(LinearOptions(in_channel, 64)))),
                                                  linear2(register_module("linear2", Linear(LinearOptions(64, 64)))),
                                                  linear3(register_module("linear3", Linear(LinearOptions(64, 64)))),
                                                  linear4(register_module("linear4", Linear(LinearOptions(64, out_channel))))
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
    Tensor r = fcnet(x); // r: [num_sample, 2], x: [num_sample, 3]

    // Vectors to multiply to Jacobian matrix
    Tensor x_vec = torch::zeros_like(r).index_put_({Slice(), Slice(0, 1)}, 1.0);
    Tensor z_vec = torch::zeros_like(r).index_put_({Slice(), Slice(1, 2)}, 1.0);

    // first order derivative 
    Tensor drdt_x = torch::autograd::grad({r}, {x}, {x_vec}, true, true)[0].set_requires_grad(true).index({Slice(), Slice(0, 1)}); // [num_sample, 1]
    Tensor drdt_z = torch::autograd::grad({r}, {x}, {z_vec}, true, true)[0].set_requires_grad(true).index({Slice(), Slice(0, 1)}); // [num_sample, 1]
    Tensor drdt = torch::cat({drdt_x, drdt_z}, 1);

    // second order derivative
    Tensor d2rdt2_x = torch::autograd::grad({drdt_x}, {x}, {torch::ones_like(drdt_x)}, true, true)[0].index({Slice(), Slice(0, 1)}); // [num_sample, ?]
    Tensor d2rdt2_z = torch::autograd::grad({drdt_z}, {x}, {torch::ones_like(drdt_z)}, true, true)[0].index({Slice(), Slice(0, 1)}); // [num_sample, ?]
    Tensor d2rdt2 = torch::cat({d2rdt2_x, d2rdt2_z}, 1);

    return {r, drdt, d2rdt2};
}
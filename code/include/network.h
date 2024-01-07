#pragma once
#include <aOpenGL.h>
#include <aLibTorch.h>
using namespace torch::indexing;


class FCImpl: public nnModule
{
public:
    Linear linear1, linear2, linear3, linear4;

    FCImpl(int in_channel, int out_channel);
    Tensor forward(Tensor x);
};
TORCH_MODULE(FC);

class GradImpl: public nnModule
{
public:
    FC fcnet;
    GradImpl(FC fcnet);
    std::tuple<Tensor, Tensor, Tensor> forward(Tensor x);
};
TORCH_MODULE(Grad);
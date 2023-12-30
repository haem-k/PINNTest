#pragma once
#include <aLibTorch.h>

class FCImpl: public nnModule
{
public:
    Linear linear1, linear2;

    FCImpl(int in_channel);
    Tensor forward(Tensor x);
};
TORCH_MODULE(FC);
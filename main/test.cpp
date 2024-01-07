#include <network.h>
#include <tensorboard_logger.h>
#include <iostream>

class Test : public agl::App
{
public:
    int nof = 50;
    int seed = 1230;
    int num_sample = 10000;

    void start() override
    {
        // TODO: Train PINN for projectile movement
        // Get device
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available())
        {
            std::cout << " * CUDA available\n"
                      << std::endl;
            device = torch::Device(torch::kCUDA);
        }
        torch::manual_seed(seed);
        torch::cuda::manual_seed(seed);

        // Set up network
        Tensor input = torch::rand({num_sample, 3}).to(device).set_requires_grad(true);
        FC fcnet = FC(3, 2);
        Grad network = Grad(fcnet);
        network->to(device);

        // Set up optimizer
        auto opt_option = torch::optim::LBFGSOptions();
        torch::optim::LBFGS optimizer = torch::optim::LBFGS(network->parameters(), opt_option);

        auto cost = [&]()
        {
            optimizer.zero_grad();

            Tensor r, drdt, d2rdt2;
            std::tie(r, drdt, d2rdt2) = network(input);
            
            std::cout << "this works??" << std::endl;
            exit(0);
            // d.backward();
            return r;
        };

        // torch::optim::Optimizer::LossClosure closure = torch::optim::Optimizer::LossClosure()
        for (int i = 0; i < 50; ++i)
        {
            optimizer.step(cost); // for LBFGS
            // cost(); optimizer.step(); // for SGD
        }
        std::cout << cost() << std::endl;
        exit(0);

        // TODO: Test PINN and render sphere
    }

    int frame = 0;
    void update() override
    {
        // Update frame count
        frame = (frame + 1) % nof;
    }

    void render() override
    {
        agl::Render::plane()
            ->scale(15.0f)
            ->floor_grid(true)
            ->color(0.2f, 0.2f, 0.2f)
            ->draw();
    }

    void key_callback(char key, int action) override
    {
        if (action != GLFW_PRESS)
            return;

        if (key == '1')
            this->capture(true);
        if (key == '2')
            this->capture(false);
    }
};

int main(int argc, char *argv[])
{
    Test app;
    agl::AppManager::start(&app);
    return 0;
}
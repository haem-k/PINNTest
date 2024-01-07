#include <network.h>
#include <tensorboard_logger.h>
#include <iostream>

class Test : public agl::App
{
public:
    int nof = 50;
    int seed = 1230;
    int num_sample = 10000;
    const float gravity = 9.8;

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
        
        Tensor t0 = torch::ones_like(input);
        t0.index_put_({Slice(), Slice(1, 3)}, 1.0);
        Tensor input_t0 = input * t0; // * 시간을 제외한 나머지 두 개의 input은 값 유지, 시간만 0으로 강제

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
            Tensor r_t0, drdt_t0, d2rdt2_t0;
            std::tie(r, drdt, d2rdt2) = network(input);
            std::tie(r_t0, drdt_t0, d2rdt2_t0) = network(input_t0);
            
            // Compute loss
            Tensor acceleration = torch::nn::functional::mse_loss(d2rdt2, torch::ones_like(d2rdt2) * -gravity);
            Tensor init_condition = torch::nn::functional::mse_loss(r_t0, torch::zeros_like(r_t0));
            Tensor loss = acceleration + init_condition;

            // Compute gradient
            loss.backward({}, true);
            return loss;
        };

        // torch::optim::Optimizer::LossClosure closure = torch::optim::Optimizer::LossClosure()
        for (int i = 0; i < 50; ++i)
        {
            optimizer.step(cost); // for LBFGS
            // cost(); optimizer.step(); // for SGD
            std::cout << i << std::endl;
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
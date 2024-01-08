#include <network.h>
#include <tensorboard_logger.h>
#include <iostream>
#include <filesystem>

class Test : public agl::App
{
public:
    int nof = 50;
    int seed = 1230;
    const int num_sample = 10000;
    const int input_feature_dim = 3;
    const int output_feature_dim = 2;
    const int num_epoch = 50;
    const float gravity = 9.8;
    const std::string test_name = "first_train";
    const std::string log_dir = "./tensorboard/first_train";
    const std::string log_file = log_dir + "/tfevents.pb";

    void check_create_dir(std::string dir)
    {
        if (agl::file_check(dir) == false)
            std::filesystem::create_directories(dir);
    }

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
        FC fcnet = FC(input_feature_dim, output_feature_dim);
        Grad network = Grad(fcnet);
        network->to(device);

        // Set up optimizer
        auto opt_option = torch::optim::LBFGSOptions();
        torch::optim::LBFGS optimizer = torch::optim::LBFGS(network->parameters(), opt_option);
        
        // Ready input
        Tensor input = torch::rand({num_sample, input_feature_dim}).to(device).set_requires_grad(true);
        Tensor t0 = torch::zeros_like(input);
        t0.index_put_({Slice(), Slice(1, 3)}, 1.0);
        Tensor input_t0 = input * t0; // * 시간을 제외한 나머지 두 개의 input은 값 유지, 시간만 0으로 강제
        
        // Ready GT
        Tensor acceleration_gt = torch::ones({num_sample, output_feature_dim}) * -gravity;
        acceleration_gt = acceleration_gt.index_put_({Slice(), Slice(0, 1)}, 0).to(device);
        Tensor init_position_gt = torch::zeros({num_sample, output_feature_dim}).to(device);
        Tensor init_velocity_gt = input.index({Slice(), Slice(1, 3)}).to(device);

        check_create_dir(log_dir);
        TensorBoardLogger train_logger(log_file.c_str());
        for (int i = 0; i < num_epoch; ++i)
        {

            auto cost = [&]()
            {
                optimizer.zero_grad();

                Tensor r, drdt, d2rdt2;
                Tensor r_t0, drdt_t0, d2rdt2_t0;
                std::tie(r, drdt, d2rdt2) = network(input);
                std::tie(r_t0, drdt_t0, d2rdt2_t0) = network(input_t0);

                // Compute loss
                Tensor acceleration = torch::nn::functional::mse_loss(d2rdt2, acceleration_gt);
                Tensor init_position = torch::nn::functional::mse_loss(r_t0, init_position_gt);
                Tensor init_velocity = torch::nn::functional::mse_loss(drdt_t0, init_velocity_gt);
                Tensor loss = acceleration + init_position + init_velocity;

                // Compute gradient
                loss.backward({}, true);
                return loss;
            };

            optimizer.step(cost); // for LBFGS
            float loss_val = cost().mean().item<float>();
            train_logger.add_scalar("train/loss", i, loss_val);
            std::cout << "[Iter: " << i << "] Loss: " << loss_val << std::endl;
        }
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
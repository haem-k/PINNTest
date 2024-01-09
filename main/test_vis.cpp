#include <network.h>
#include <tensorboard_logger.h>
#include <iostream>
#include <filesystem>

class Test : public agl::App
{
public:
    const std::string test_name = "first_train";
    const std::string log_dir = "./tensorboard/" + test_name;
    const std::string log_file = log_dir + "/tfevents.pb";
    const std::string model_dir = "./model/" + test_name;

    const int seed = 1230;
    const int num_test_sample = 100;
    const int input_feature_dim = 3;
    const int output_feature_dim = 2;
    const int num_epoch = 100;

    Tensor output_r;

    int nof;
    std::string frame_text;
    std::string coord_text;
    Vec3 frame_text_position = Vec3(0, 0.5, 0);
    Vec3 coord_text_position = Vec3(0, 0.2, 0);
    Vec3 sphere_position = Vec3().Zero();

    void check_create_dir(std::string dir)
    {
        if (agl::file_check(dir) == false)
            std::filesystem::create_directories(dir);
    }

    void start() override
    {
        // Test PINN for projectile motion
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
        torch::load(network, model_dir + "/model.pt");
        network->to(device);
        network->eval();

        // Ready input
        Tensor input_t = torch::linspace(0, 1, num_test_sample).reshape({num_test_sample, 1});
        Tensor input_v0 = 0.5 * torch::ones({num_test_sample, 2});
        Tensor input = torch::cat({input_t, input_v0}, 1).to(device).set_requires_grad(true);

        Tensor drdt, d2rdt2;
        std::tie(output_r, drdt, d2rdt2) = network(input); // output_r: [100, 2] --> 사실 Test 할 때는 미분할 필요가 없구나

        nof = num_test_sample;
    }

    bool stop = false;
    int frame = 0;
    void update() override
    {
        frame_text = "Frame " + std::to_string(frame + 1);

        // Render sphere
        sphere_position.x() = output_r[frame][0].item<float>();
        sphere_position.y() = output_r[frame][1].item<float>() + 5;
        coord_text = "x = " + std::to_string(sphere_position.x()) + "  y = " + std::to_string(sphere_position.y());

        if (stop)
            return;

        // Update frame count
        frame = (frame + 1) % nof;
    }

    void render() override
    {
        agl::Render::plane()
            ->scale(15.0f)
            ->floor_grid(true)
            ->color(0.1f, 0.1f, 0.1f)
            ->draw();

        agl::Render::sphere()
            ->scale(0.1f)
            ->position(sphere_position)
            ->color(1.0f, 0.0f, 0.0f)
            ->draw();

        agl::Render::text(frame_text, 1.0f)
            ->scale(0.9f)
            ->position(frame_text_position)
            ->draw();

        agl::Render::text(coord_text, 1.0f)
            ->scale(0.9f)
            ->position(coord_text_position)
            ->draw();
    }

    void key_callback(char key, int action) override
    {
        if (action != GLFW_PRESS)
            return;
        if (key == 's')
            stop = !stop;
        if (key == 'q')
        {
            frame--;
            if (frame < 0)
                frame += nof;
        }
        if (key == 'w')
        {
            frame = (frame + 1) % nof;
        }
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
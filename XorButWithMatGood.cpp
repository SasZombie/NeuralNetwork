#include "NeuralNetwork.hpp"


float td[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0
};


int main()
{

    size_t arch[] = {2, 2, 1};

    size_t stride = 3;

    size_t n = sizeof(td)/sizeof(td[0])/stride;
    
    nn::NN nn(arch, 3);
    nn::NN grad(arch, 3);


    nn::Mat ti(4, 2,stride, td);
    nn::Mat to(4, 1,stride, td+2);
    nn.rand();
   
    float rate = 1;


    //std::cout << nn.cost(ti, to) << '\n';


    for(size_t i = 0; i < 5000; ++i)
    {   
        nn.backProp(grad, ti, to);   
        nn.learn(grad, rate);
        std::cout << "\nCost = " << nn.cost(ti, to) << '\n' ;
    }

    for (size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
        {
            nn.setAtAs(0, 0, 0, i);
            nn.setAtAs(0, 0, 1, j);
            nn.forward();
            std::cout << i << " ^ " << j << " => " << nn.getOutput().getAt(0, 0) << '\n';
        }
    }
}
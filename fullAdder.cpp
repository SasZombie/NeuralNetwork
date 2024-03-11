#include "NeuralNetwork.hpp"
#include <bitset>

constexpr size_t BITS = 2;

int main()
{
    size_t n = (1 << BITS);
    size_t rows = n + n;
    nn::Mat ti(rows, BITS+BITS);
    nn::Mat to(rows, BITS+1);

    for (size_t i = 0; i < ti.getRows(); i++)
    {
        std::bitset<BITS+BITS> curNrr(i);

        for (size_t j = 0; j < ti.getCols(); j++)
        {
            ti.setAt(i, j, curNrr[j]);   
        }
        std::bitset<BITS>ls;
        std::bitset<BITS>rs;


        for (size_t j = 0; j < curNrr.size()/2; ++j)
        {
            ls.set(j, curNrr[j]);
            rs.set(j, curNrr[j+BITS]);
        }
        
        curNrr = ls.to_ulong() + rs.to_ulong();
        for (size_t j = 0; j < to.getCols(); j++)
        {
            
            to.setAt(i, j, curNrr[j]);
        }
    }


    size_t arch[]= {2 * BITS, 2 * BITS + 1, BITS+1};
    size_t len = sizeof(arch)/sizeof(arch[0]);
    nn::NN nn(arch, len);
    nn::NN grad(arch, len);

    nn.rand();

    float rate = 1;

    std::cout << nn.cost(ti, to) << ' ';
    for (size_t i = 0; i < 10 * 1000; ++i)
    {
        //nn.fineDiff(grad, 1e-1, ti, to);
        nn.backProp(grad, ti, to);
        nn.learn(grad, rate);
       
        std::cout << nn.cost(ti, to) << '\n';
    }
    
    //nn.print();
}
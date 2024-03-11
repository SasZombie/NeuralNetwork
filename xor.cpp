#include <iostream>
#include <random>
#include <iomanip>


struct Xor
{
    float or_w1;
    float or_w2;
    float or_b;
 
    float nand_w1;
    float nand_w2;
    float nand_b;
    
    float and_w1;
    float and_w2;
    float and_b;
};


float forward(Xor m, float x, float y);
float getRandom();
float sigmoid(float x);
float cost(Xor m);

Xor randomXor();
Xor finate_diff(Xor m, float esp);

Xor apply_diff(Xor m, Xor g, float rate);

void print_xor(Xor m);



using sample = float[3];

sample train_or[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
};

//And gate
sample train_and[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1}
};

//Nand gate
sample train_nand[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}
};

//Xor
sample train_xor[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}
};

const int train_count = 4;

sample *train = train_xor;


float getRandom()
{
    std::random_device rd;

    std::default_random_engine engine(rd());

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}

float sigmoid(float x)
{
    return 1.f/(1.f + std::exp(-x));
}

float forward(Xor m, float x, float y)
{   

    float a = sigmoid(m.or_w1*x + m.or_w2 * y + m.or_b);

    float b = sigmoid(m.nand_w1*x + m.nand_w2 * y + m.nand_b);

    return sigmoid(m.and_w1 * a + m.and_w2 * b + m.and_b);

}


float cost(Xor m)
{
    float result = 0.0f;

    for (size_t i = 0; i < train_count; ++i)
    {
        float x0 = train[i][0];
        float x1 = train[i][1];


        //std::cout << x0 << ' ' << x1 << '\n';
        float y = forward(m, x0, x1);
        //std::cout <<"Final " <<  y << ' ';
        float d = y - train[i][2];

        result = result + d*d;
    }

    result = result/train_count;

    return result;   
}

Xor randomXor()
{
    Xor m;

    m.or_w1 = getRandom();
    m.or_w2 = getRandom();
    m.or_b = getRandom();

    m.nand_w1 = getRandom();
    m.nand_w2 = getRandom();
    m.nand_b = getRandom();

    m.and_w1 = getRandom();
    m.and_w2 = getRandom();
    m.and_b = getRandom();

   

    return m;
}

void print_xor(Xor m)
{
    std::cout << m.or_w1 << '\n' << m.or_w2 << '\n' << m.or_b << '\n'<< m.and_w1 << '\n'
                    << m.and_w2 << '\n' << m.and_b << '\n'<< m.nand_w1 << '\n' << m.nand_w2
                                         << '\n' << m.nand_b << '\n';
}

Xor finate_diff(Xor m, float esp)
{
    Xor g;
    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += esp;
    g.or_w1 = (cost(m)-c)/esp;
    m.or_w1 = saved;


    saved = m.or_w2;
    m.or_w2 += esp;
    g.or_w2 = (cost(m)-c)/esp;
    m.or_w2 = saved;


    saved = m.or_b;
    m.or_b += esp;
    g.or_b = (cost(m)-c)/esp;
    m.or_b = saved;


    saved = m.nand_w1;
    m.nand_w1 += esp;
    g.nand_w1 = (cost(m)-c)/esp;
    m.nand_w1 = saved;


    saved = m.nand_w2;
    m.nand_w2 += esp;
    g.nand_w2 = (cost(m)-c)/esp;
    m.nand_w2 = saved;


    saved = m.nand_b;
    m.nand_b += esp;
    g.nand_b = (cost(m)-c)/esp;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += esp;
    g.and_w1 = (cost(m)-c)/esp;
    m.and_w1 = saved;


    saved = m.and_w2;
    m.and_w2 += esp;
    g.and_w2 = (cost(m)-c)/esp;
    m.and_w2 = saved;


    saved = m.and_b;
    m.and_b += esp;
    g.and_b = (cost(m)-c)/esp;
    m.and_b = saved;

   

    return g;
}

Xor apply_diff(Xor m, Xor g, float rate)
{

    std::cout <<  m.or_w1 - rate * g.or_w1; 
    m.or_w1 -= rate * g.or_w1; 
    m.or_w2 -= rate * g.or_w2; 
    m.or_b -= rate * g.or_b; 

    m.nand_w1 -= rate * g.nand_w1; 
    m.nand_w2 -= rate * g.nand_w2; 
    m.nand_b -= rate * g.nand_b; 


    m.and_w1 -= rate * g.and_w1; 
    m.and_w2 -= rate * g.and_w2; 
    m.and_b -= rate * g.and_b; 

    return m;
    
}

int main()
{


    Xor m = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    float eps = 1e-1;
    
    float rate = 1e-1;
    Xor g = finate_diff(m, eps);

    
    m = apply_diff(m, g, rate);

    print_xor(m);


#if 0

    for (size_t i = 0; i < 100 * 1000; ++i)
    {


    }

    for (size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
            std::cout << i << " ^ " << j << " => " << forward(m, i, j) << '\n';
    }
    
#endif
    // Chat GBT => 1 000 000 000 000
    // Me => 3
}

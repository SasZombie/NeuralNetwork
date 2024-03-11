#include "NeuralNetwork.hpp"

struct Xor
{
    nn::Mat a0, a1, a2;

    nn::Mat w1, b1;
    
    nn::Mat w2, b2;

};

void forward_xor(Xor& model);
float cost(Xor m, nn::Mat ti, nn::Mat to);
void fine_diff(Xor& m, Xor& g, float eps, nn::Mat ti, nn::Mat to);
void xor_alloc(Xor& x);
void learn(Xor& m, Xor& g, float rate);

void forward_xor(Xor& model)
{
    model.a1.dot(model.a0, model.w1);
    model.a1.sum(model.b1);
    model.a1.apply_sigmoid();

    model.a2.dot(model.a1, model.w2);
    model.a2.sum(model.b2);
    model.a2.apply_sigmoid();

}

void fine_diff(Xor& m, Xor& g, float eps, nn::Mat ti, nn::Mat to)
{
    float saved;
    float c = cost(m, ti, to);
    for(size_t i = 0; i < m.w1.getRows(); ++i)
    {
        for(size_t j = 0; j < m.w1.getCols(); ++j)
        {
            saved = m.w1.getAt(i, j);
            float temp1 = saved + eps, temp2 = (cost(m, ti, to) - c) / eps;

            m.w1.setAt(i, j, temp1);
            g.w1.setAt(i, j, temp2);

            m.w1.setAt(i, j, saved);
        }
    }

    for(size_t i = 0; i < m.b1.getRows(); ++i)
    {
        for(size_t j = 0; j < m.b1.getCols(); ++j)
        {
            saved = m.b1.getAt(i, j);
            float temp1 = saved + eps, temp2 = (cost(m, ti, to) - c) / eps;

            m.b1.setAt(i, j, temp2);
            g.b1.setAt(i, j, temp1);

            m.b1.setAt(i, j, saved);
        }
    }

    for(size_t i = 0; i < m.w2.getRows(); ++i)
    {
        for(size_t j = 0; j < m.w2.getCols(); ++j)
        {
            saved = m.w2.getAt(i, j);
            float temp1 = saved + eps, temp2 = (cost(m, ti, to) - c) / eps;
            m.w2.setAt(i, j, temp1);
            g.w2.setAt(i, j, temp2);

            m.w2.setAt(i, j, saved);
        }
    }

    for(size_t i = 0; i < m.b2.getRows(); ++i)
    {
        for(size_t j = 0; j < m.b2.getCols(); ++j)
        {
            saved = m.b2.getAt(i, j);
            float temp1 = saved + eps, temp2 = (cost(m, ti, to) - c) / eps;
            m.b2.setAt(i, j, temp1);
            g.b2.setAt(i, j, temp2);

            m.b2.setAt(i, j, saved);
        }
    }

}

float cost(Xor m, nn::Mat ti, nn::Mat to)
{
    assert(ti.getRows() == to.getRows());
    assert(to.getCols() == m.a2.getCols());

    size_t n = ti.getRows();
    float c = 0;

    for (size_t i = 0; i < n; ++i)
    {
        nn::Mat x = ti.matRow(i);
        nn::Mat y = to.matRow(i);
        m.a0 = x;

        forward_xor(m);
        
        size_t q = to.getCols();

        for (size_t j = 0; j < q; ++j)
        {
            float d = m.a2.getAt(0, j) - y.getAt(0, j);
            c = c + d*d;
        }

    }
    return c/n;
}

void xor_alloc(Xor& model)
{
    model.a0.alloc(1, 2);
    
    model.w1.alloc(2, 2);
    model.b1.alloc(1, 2);
    model.a1.alloc(1, 2);
    
    model.w2.alloc(2, 1);
    model.b2.alloc(1, 1);
    model.a2.alloc(1, 1);
}

void learn(Xor& m, Xor& g, float rate)
{
    for(size_t i = 0; i < m.w1.getRows(); ++i)
    {
        for(size_t j = 0; j < m.w1.getCols(); ++j)
        {
            float temp = (m.w1.getAt(i, j) - g.w1.getAt(i, j)) * rate;
            m.w1.setAt(i, j, temp);
        }
    }

    for(size_t i = 0; i < m.b1.getRows(); ++i)
    {
        for(size_t j = 0; j < m.b1.getCols(); ++j)
        {
            float temp = (m.b1.getAt(i, j) - g.b1.getAt(i, j)) * rate;
            m.b1.setAt(i, j, temp);
        }
    }

    for(size_t i = 0; i < m.w2.getRows(); ++i)
    {
        for(size_t j = 0; j < m.w2.getCols(); ++j)
        {
            float temp = (m.w2.getAt(i, j) - g.w2.getAt(i, j)) * rate;
            m.w2.setAt(i, j, temp);
        }
    }

    for(size_t i = 0; i < m.b2.getRows(); ++i)
    {
        for(size_t j = 0; j < m.b2.getCols(); ++j)
        {
            float temp = (m.b2.getAt(i, j) - g.b2.getAt(i, j)) * rate;
            m.b2.setAt(i, j, temp);
        }
    }

}
float td[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0
};

int main()
{
    Xor model;
    Xor gradient;
    
    size_t stride = 3;

    size_t n = sizeof(td)/sizeof(td[0])/stride;

    nn::Mat ti(n, 2, stride, td);

    nn::Mat to(n, 1, stride, td+2);

    ti.print();
    to.print();

    xor_alloc(model);
    xor_alloc(gradient);
  

    model.w1.rand();
    model.w2.rand();
    model.b1.rand();
    model.b2.rand();

#if 1
  //  std::cout << "Cost = " << cost(model, ti, to) << '\n';
    
    
    float eps = 1e-1;
    float rate = 1e-1;
    for (size_t i = 0; i < 10 ; ++i)
    {
        fine_diff(model, gradient, eps, ti, to);
        learn(model, gradient, rate);

        std::cout << "Cost2 = " << cost(model, ti, to) << '\n';
    }
    
   
#endif

#if 0
    int a = 2;

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            model.a0.setAt(0, 0, i);
            model.a0.setAt(0, 1, j);

            forward_xor(model);

            float y = model.a2.getAt(0, 0);

            std::cout << i << " ^ " << j << " => " << y << " -> Expected was: "<< td[a]  << '\n';
            a = a + 3;
        }
        
    }
#endif

}
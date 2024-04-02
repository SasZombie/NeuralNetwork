#include "NeuralNetwork.hpp"

void nn::Mat::setEs(const float* n_es)
{
    //std::copy(n_es, n_es + rows * stride, es.begin());

    this->empty = false;
    for(size_t i = 0; i < cols; ++i)
    {
        this->es[i] = n_es[i];
    }
}


size_t nn::Mat::getRows() const noexcept
{
    return rows;
}

size_t nn::Mat::getCols() const noexcept
{
    return cols;
}

void nn::Mat::alloc(const size_t n_rows, const size_t n_cols)
{
    this->empty = true;
    this->rows = n_rows;
    this->cols = n_cols;
    this->stride = n_cols;

    this->es.resize(rows * cols);
     
}

void nn::Mat::alloc(const size_t n_rows, const size_t n_cols, const size_t n_stride)
{
    this->rows = n_rows;
    this->cols = n_cols;
    this->stride = n_stride;

    this->es.resize(rows * cols);
    
}

nn::Mat nn::Mat::operator+(const Mat &other) const noexcept
{
    
    assert(this->cols == other.cols);
    size_t toAddRows, otherToAddRows;


    if(!this->empty)
        toAddRows = this->rows;
    else
        toAddRows = 0;

    if(!other.empty)
        otherToAddRows = other.rows;
    else
        otherToAddRows = 0;

    size_t newRows = toAddRows + otherToAddRows;
    assert(newRows>0);
    Mat retMat{newRows, this->cols};

    retMat.empty = false;

    if(this->empty)
    {
        for(size_t i = 0; i < other.rows; ++i)
        {
            for(size_t j = 0; j < cols; ++j)
            {
                mat_at_non(retMat, i, j) = mat_at_non(other, i, j);
            }
        }
    }
    else if(other.empty)
    {
        for(size_t i = 0; i < rows; ++i)
        {
            for(size_t j = 0; j < cols; ++j)
            {
                mat_at_non(retMat, i, j) = mat_at(this, i, j);
            }
        }
    }
    else
    {
      
        for(size_t i = 0; i < rows; ++i)
        {
            for(size_t j = 0; j < cols; ++j)
            {
                mat_at_non(retMat, i, j) = mat_at(this, i, j);
            }
        }

        for(size_t i = 0; i < other.rows; ++i)
        {
            for(size_t j = 0; j < cols; ++j)
            {
                mat_at_non(retMat, i + rows, j) = mat_at_non(other, i, j);
            }
        }
    }

    return retMat;
}


nn::Mat nn::Mat::matRow(size_t row) const
{
    Mat m(1, cols);  
    for(size_t i = 0; i < cols; ++i)
    {
        m.setAt(0, i, this->getAt(row, i));
    }

    return m;  
}

void nn::Mat::rand(const float low, const float max) noexcept
{
    this->empty = false;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) = getRandom() * (max - low) + low;
        }
    }
}

void nn::Mat::fill(const float x) noexcept
{
    this->empty = false;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) = x;
        }
    }
}

void nn::Mat::dot(const Mat a, const Mat b)
{
    assert(a.cols == b.rows);
    size_t n = a.cols;
    assert(this->cols == b.cols);
    assert(this->rows == a.rows);
    
    
    for (size_t i = 0; i < this->rows; ++i)
    {
        for (size_t j = 0; j < this->cols; ++j)
        {
            for (size_t k = 0; k < n; ++k)
            {
                mat_at(this, i, j) += mat_at_non(a, i, k) * mat_at_non(b, k, j);
            }
        }        
    }    
    
}

void nn::Mat::save(std::ofstream &file) const noexcept
{
    const std::string magic = "susmat";
    
    file.write(magic.c_str(), magic.size());
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

    file.write(reinterpret_cast<const char*>(es.data()), cols * rows*sizeof(float));

}

void nn::Mat::load(std::ifstream &file) noexcept
{
    const std::string expectedMagic("susmat");

    char buffer[6];
    file.read(buffer, sizeof(buffer));

    std::string buff(buffer, sizeof(buffer));
    if(buff != expectedMagic)
    {
        std::cerr << buff << " is not a valid .mat file\n";
        return;
    }


    file.read(reinterpret_cast<char*>(&this->rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&this->cols), sizeof(size_t));

    if(this->rows < 1 || this->cols < 1)
    {
        std::cerr << "Matrix cannot have 0 or negative row or col values\n";
        return;
    }

    this->es.resize(this->rows * this->cols);

    file.read(reinterpret_cast<char*>(es.data()), this->rows * this->cols * sizeof(float));

    this->stride = this->cols;
    
}


void nn::Mat::sum(const Mat a)
{
    assert(this->rows == a.rows);
    assert(this->cols == a.cols);
    
    for (size_t i = 0; i < a.rows; ++i)
    {
        for (size_t j = 0; j < a.cols; ++j)
        {
            mat_at(this, i, j) = mat_at(this, i, j ) + mat_at_non(a, i, j);
        }        
    }
}

void nn::Mat::shuffle() noexcept
{
    for (size_t i = 0; i < rows; ++i)
    {
        size_t rand = getRandom() * (rows - i) + i;
        for (size_t j = 0; j < cols; ++j)
        {
            float t = mat_at(this, rand, j);
            mat_at(this, rand, j) = mat_at(this, i, j);
            mat_at(this, i, j) = t;
        }
    }
}

void nn::Mat::append(const Mat &m) noexcept
{
    this->empty = false;
    size_t newCols = this->cols + m.cols;

    std::vector<float> combine(rows * newCols);  

    for(size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            combine[i*stride + j] = mat_at(this, i, j);
        }  
    }

    for(size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            combine[i*stride + (j+cols)] = mat_at_non(m, i, j);
        }  
    }

    this->cols = newCols;
    this->es = combine;
 
}

void nn::Mat::print(const std::string& name) const noexcept
{
    std::cout << "------------\n";
    std::cout << name << '\n';

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << mat_at(this, i, j) << " ";
        }
        std::cout << '\n';   
    }

}

void nn::Mat::clear() noexcept
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) = 0;
    
        }
    }
}

void nn::Mat::setAt(size_t i, size_t j, float number) noexcept
{
    this->empty = false;
    // std::cout << "size=" << this->es.size() << "sum = " << rows * cols << "How much I have " << i*stride + j << "i = " << i << " j = " << j << '\n';
    mat_at(this, i, j) = number;
}

void nn::Mat::setAt(const size_t i, const float number) noexcept
{
    this->empty = false;
    es[i] = number;
}

nn::Mat nn::Mat::slice(size_t row, size_t startPos, size_t n_cols) noexcept
{
    nn::Mat mat(1, n_cols);

    std::vector<float> newVec;

    for (size_t j = 0; j < n_cols; ++j)
    {
        newVec.push_back(mat_at(this, row, (j + startPos)));    
    }
    
    mat.setEs(newVec.data());

    return mat;
}

float nn::Mat::getAt(const size_t i) noexcept
{
    return es[i];
}

float nn::Mat::getAt(const size_t i, const size_t j) const noexcept
{
    return mat_at(this, i, j);
}

void nn::Mat::apply_sigmoid() noexcept
{
    for(size_t i = 0; i < rows; ++i)
    {
        for(size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) = sig(mat_at(this, i, j));
        }
    }
}

void nn::Mat::apply_relu() noexcept
{
    for(size_t i = 0; i < rows; ++i)
    {
        for(size_t j = 0; j < cols; ++j)
        {
            mat_at(this, i, j) = relu(mat_at(this, i, j));
        }
    }
}


float nn::sig(const float x) noexcept
{
    return 1.f/(1.f + std::exp(-x));
}

float nn::relu(const float x) noexcept
{
    if(x > 0)
        return x;
    return 0;
}

float nn::Mat::getRandom() const noexcept
{
    static std::random_device rd;

    static std::default_random_engine engine(rd());

    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}

size_t nn::Mat::getStride() const noexcept
{
    return this->stride;
}

const float *nn::Mat::getData() const noexcept
{
    return this->es.data();
}


nn::NN::NN(size_t *arch, size_t arch_count)
{
    if(arch_count < 1)
    {
        std::cout << "Arch count cannot be 0";
        return;
    }
    
    this->count = arch_count - 1;

    this->ws.resize(this->count);
   
    this->bs.resize(this->count);

    this->as.resize(this->count + 1);

    this->as[0].alloc(1, arch[0]);


    for (size_t i = 1; i < arch_count; ++i)
    {
        this->ws[i-1].alloc(this->as[i - 1].getCols(), arch[i]);
        this->bs[i-1].alloc(1, arch[i]);
        this->as[i].alloc(1, arch[i]);

    }
}

nn::NN::NN(const std::string &file)
{
    std::ifstream input(file, std::ios::binary);

    load(input);
}

void nn::NN::print() const noexcept
{
    for (size_t i = 0; i < this->count ; ++i)
    {
        std::string s = "ws[" + std::to_string(i) + "] = ";
        this->ws[i].print(s);
        s.clear();
        s = "bs[" + std::to_string(i) + "] = ";
        this->bs[i].print(s);
    }
    
}

void nn::NN::alloc(size_t *arch, size_t arch_count)
{
   if(arch_count < 1)
    {
        std::cout << "Arch count cannot be 0";
        return;
    }
    
    this->count = arch_count - 1;

    this->ws.resize(this->count);
   
    this->bs.resize(this->count);

    this->as.resize(this->count + 1);

    this->as[0].alloc(1, arch[0]);



    this->as[0].alloc(1, arch[0]);


    for (size_t i = 1; i < arch_count; ++i)
    {
        this->ws[i-1].alloc(this->as[i - 1].getCols(), arch[i]);
        this->bs[i-1].alloc(1, arch[i]);
        this->as[i].alloc(1, arch[i]);

    }
}

void nn::NN::rand(const float low, const float max)
{
    for (size_t i = 0; i < count; ++i)
    {
        this->ws[i].rand(low, max);
        this->bs[i].rand(low, max);
    }
    
}

void nn::NN::setActivation(activations activation)
{
    this->actFunction = activation;
}

void nn::NN::clear() noexcept
{
    for(size_t i = 0; i < count; ++i)
    {
        this->ws[i].clear();
        this->bs[i].clear();
        this->as[i].clear();
    }

    this->as[count].clear();
}

nn::Mat& nn::NN::getInput() noexcept
{
    return this->as[0];
}

nn::Mat& nn::NN::getOutput() noexcept
{
    return this->as[count];
}

void nn::NN::setInput(const nn::Mat &m) noexcept
{
    this->as[0] = m;
}

void nn::NN::setOutput(const nn::Mat &m) noexcept
{
    this->as[this->count] = m;
}

size_t nn::NN::getCount() const noexcept
{
    return count;
}

void nn::NN::forward() noexcept
{
    for (size_t i = 0; i < count; ++i)
    {   
        this->as[i+1].clear();
        this->as[i+1].dot(this->as[i], this->ws[i]);
        this->as[i+1].sum(this->bs[i]);

        switch (this->actFunction)
        {
        case activations::sigmoid :
            this->as[i+1].apply_sigmoid();
            break;
        case activations::relu: 
            this->as[i+1].apply_relu();
            break;
        default:
            break;
        }

    }
    
}

void nn::NN::backProp(NN &grad, const Mat &ti, const Mat &to)
{
    assert(ti.getRows() == to.getRows());
    size_t n = ti.getRows();
    assert(this->getOutput().getCols() == to.getCols());
    grad.clear();

    for(size_t i = 0; i < n; ++i)
    {
        this->setInput(ti.matRow(i));
        this->forward();
        
        for(size_t j = 0; j <= count; ++j)
        {
            grad.as[j].clear();
        }

        for(size_t j = 0; j < to.getCols(); ++j)
        {
            float item = this->getOutput().getAt(0, j) - to.getAt(i, j);
            grad.getOutput().setAt(0, j, item);

        }

        // std::cout << "Count = " << this->as[this->count].getCols() << '\n';

        for(size_t l = this->count; l > 0; --l)
        {

            for(size_t j = 0; j < this->as[l].getCols(); ++j)
            {


                float a = this->as[l].getAt(0, j);
            
                float da = grad.as[l].getAt(0, j);

                // std::cout << "j = " << j << '\n';

                float q;
                if(this->actFunction == activations::sigmoid)
                    q = a * (1-a);
                else
                    q = (a > 0);
                
                grad.setAtBs(l-1, 0, j, (grad.getAtBs(l-1, 0, j) + 2 * da * q));

                for(size_t k = 0; k < this->as[l-1].getCols(); ++k)
                {

                                   
                    float pa = this->as[l-1].getAt(0, k);
                    float w = this->ws[l-1].getAt(k, j);

                    grad.setAtWs(l-1, k, j, grad.getAtWs(l-1, k, j) + 2* da * q * pa);

                    grad.setAtAs(l-1, 0, k, grad.getAtAs(l-1, 0, j) + 2* da * q * w);

                }
            }
        }
    }

    for (size_t i = 0; i < grad.getCount(); ++i)
    {
        for (size_t j = 0; j < grad.ws[i].getRows(); ++j)
        {
            for (size_t k = 0; k < grad.ws[i].getCols(); ++k)
            {
                grad.setAtWs(i, j, k, grad.getAtWs(i, j, k)/n);
            }
        }
        
        for (size_t j = 0; j < grad.bs[i].getRows(); ++j)
        {
            for (size_t k = 0; k < grad.bs[i].getCols(); ++k)
            {
                grad.setAtBs(i, j, k, grad.getAtBs(i, j, k)/n);
            }
        }
        
    }

}



void nn::NN::fineDiff(NN &grad, const float eps, const Mat &ti, const Mat &to)
{
    float saved;
    float c = cost(ti, to);

    for(size_t i = 0; i < count; ++i)
    {
        for(size_t j = 0; j < this->ws[i].getRows(); ++j)
        {
            for(size_t k = 0; k < this->ws[i].getCols(); ++k)
            {


                saved = this->ws[i].getAt(j, k);

                this->ws[i].setAt(j, k, saved + eps);

                grad.setAtWs(i, j, k, (this->cost(ti, to) - c)/eps);
                
                this->ws[i].setAt(j, k, saved);
            }
        }

        for(size_t j = 0; j < this->bs[i].getRows(); ++j)
        {
            for(size_t k = 0; k < this->bs[i].getCols(); ++k)
            {
                saved = this->bs[i].getAt(j, k);

                this->bs[i].setAt(j, k, saved + eps);
       
                grad.setAtBs(i, j, k, (this->cost(ti, to) - c)/eps);
                
                this->bs[i].setAt(j, k, saved);
            }
        }
    }
}

void nn::NN::learn(const NN &grad, float rate)
{
    for(size_t i = 0; i < count; ++i)
    {
        for(size_t j = 0; j < this->ws[i].getRows(); ++j)
        {
            for(size_t k = 0; k < this->ws[i].getCols(); ++k)
            {
                this->ws[i].setAt(j, k, this->ws[i].getAt(j, k) - grad.getAtWs(i, j, k) * rate);
            }
        }

        for(size_t j = 0; j < this->bs[i].getRows(); ++j)
        {
            for(size_t k = 0; k < this->bs[i].getCols(); ++k)
            {
                this->bs[i].setAt(j, k, this->bs[i].getAt(j, k) - grad.getAtBs(i, j, k) * rate);
            }
        }
    }
}

float nn::NN::cost(const Mat &ti, const Mat &to)
{
    assert(ti.getRows() == to.getRows());
    assert(to.getCols() == this->getOutput().getCols());

    const size_t n = ti.getRows();
    const size_t q = to.getCols();
    float c = 0.0f;
 
    for (size_t i = 0; i < n; ++i)
    {
        const nn::Mat x = ti.matRow(i);
        const nn::Mat y = to.matRow(i);

        this->setInput(x);
        this->forward();

        for (size_t j = 0; j < q; ++j)
        {
            float d = this->getOutput().getAt(0, j) - y.getAt(0, j);
            c = c + d*d;
        }

    }

    c = c / n;
    return c;
}

void nn::NN::save(std::ofstream &path) const noexcept
{
    // if(path.flags() & static_cast<std::ios_base::fmtflags>(std::ios::app))
    // {
    //     std::cerr<<"Cannot save NN since the append file is not set!\n";

    //     return;
    // }
    const std::string magic = "susnn";
    
    path.write(magic.c_str(), magic.size());

    path.write(reinterpret_cast<const char*>(&count), sizeof(size_t));

    for(const Mat& mat : this->ws)
        mat.save(path);

    for(const Mat& mat : this->bs)
        mat.save(path);

    for(const Mat& mat : this->as)
        mat.save(path);

    std::cout << "Saved" << '\n';
}

void nn::NN::load(std::ifstream &path) noexcept
{

    const std::string expectedMagic("susnn");

    char buffer[5];
    path.read(buffer, sizeof(buffer));

    std::string buff(buffer, sizeof(buffer));
    if(buff != expectedMagic)
    {
        std::cerr << buff << " is not a valid nerutal network file\n";
        return;
    }


    path.read(reinterpret_cast<char*>(&this->count), sizeof(size_t));
    
    if(this->count < 1)
    {
        std::cerr << "Matrix cannot have 0 or negative count numbers\n";
        return;
    }

    this->ws.resize(this->count);
   
    this->bs.resize(this->count);

    this->as.resize(this->count + 1);

    for(size_t i = 0; i < count; ++i)
    {
        this->ws.at(i).load(path);
    }

    for(size_t i = 0; i < count; ++i)
    {
        this->bs.at(i).load(path);
    }
 
    for(size_t i = 0; i < count + 1; ++i)
    {
        this->as.at(i).load(path);
    }   
}

float nn::NN::getAtWs(const size_t i, const size_t j, const size_t k) const noexcept
{
    return this->ws[i].getAt(j, k);
}

void nn::NN::setAtWs(const size_t i, const size_t j, const size_t k, const float number) noexcept
{
    this->ws[i].setAt(j, k, number);
}

float nn::NN::getAtBs(const size_t i, const size_t j, const size_t k) const noexcept
{
    return this->bs[i].getAt(j, k);
}

void nn::NN::setAtBs(const size_t i, const size_t j, const size_t k, const float number) noexcept
{
    this->bs[i].setAt(j, k, number);
}

float nn::NN::getAtAs(const size_t i, const size_t j, const size_t k) const noexcept
{
    return this->as[i].getAt(j, k);
}

float nn::NN::getWsCols(const size_t i) const noexcept
{
    return this->ws[i].getCols();
}

float nn::NN::getAsCols(const size_t i) const noexcept
{
    return this->as[i].getCols();
}

void nn::NN::setAtAs(const size_t i, const size_t j, const size_t k, const float number) noexcept
{
    this->as[i].setAt(j, k, number);
}

nn::NN::~NN() = default;
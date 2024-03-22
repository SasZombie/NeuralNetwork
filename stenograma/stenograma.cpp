#include <SFML/Audio.hpp>
#include <fftw3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>

using namespace std::complex_literals;



std::vector<double> fft(const std::vector<double>& input) {
    int size = input.size();
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);

    fftw_plan plan = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < size; ++i) {
        in[i][0] = input[i];
        in[i][1] = 0.0;
    }

    fftw_execute(plan);

    std::vector<double> result(size);
    for (int i = 0; i < size; ++i) {
        result[i] = 20 * log10(sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]));
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
}


void fft(double in2[], size_t stride, std::complex<double>out2[], size_t n)
{
    if( n == 1)
    {
        out2[0] = in2[0];
        return;
    }
    fft(in2, stride*2, out2, n/2);

    fft(in2 + stride, stride*2, out2 + n/2, n/2);

    for (size_t k = 0; k < n/2; ++k)
    {
        double t = (double) k/n;
        std::complex<double> v = std::exp((-2) * M_PI * t * 1i) * out2[k+n/2];
        std::complex<double> e = out2[k];
        out2[k] = e + v;
        out2[k+n/2] = e-v;
    } 
}

void soundToMag(std::complex<double> in[], double out[], size_t N)
{
    for(size_t i = 0; i < N; ++i)
        out[i] = 20 * log10(sqrt(in->real() * in->real() + in->imag() * in->imag()));
}



int main() {
    constexpr float freqNum = 32768.f;
    sf::SoundBuffer buffer;

    if (!buffer.loadFromFile("Baldurs.wav")) 
    {
        std::cerr << "Failed to load audio file\n";
        return 1;
    }

    const sf::Int16* samples = buffer.getSamples();
    
    size_t sampleCount = buffer.getSampleCount();
    std::cout << sampleCount << '\n';
    

    unsigned int sampleRate = buffer.getSampleRate();


    std::vector<double> audioData(sampleCount);
    for (size_t i = 0; i < sampleCount; ++i) {
        audioData[i] = static_cast<double>(samples[i]) / freqNum ;
    }

    const int N = sampleCount;

    double* in = new double[N];
    in = audioData.data();
    std::complex<double> *out = new std::complex<double>[N];
    
    fft(in, 1, out, N);
    soundToMag(out, in, N);

    std::vector<double> result = fft(audioData);

    std::cout << result.size();

    size_t width = 28;
    size_t height = 28; 
    unsigned char* image = new unsigned char[width * height];
    for (size_t i = 0; i < width; ++i)
    {
        for (size_t j = 0; j < height; ++j)
        {
            double value = in[i];
            int index = j * width + i;
            if(value < 0)
            {
                image[index] = 0;

            }else if (value > 255)
            {
                image[index] = 255;
            }
            else
            {
                image[index] = static_cast<unsigned char>(value);
            }
        }
    }

    stbi_write_png("stenogramaMyFFT.png", width, height, 1, image, width);

}

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "NeuralNetwork.hpp"


std::vector<std::string> split(const std::string &s, char delimiter) 
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

int main()
{

    std::string generas[] = {
        "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"
    };
    std::ifstream file("PythonNN/Data/features_3_sec.csv", std::ios::binary); 
    if (!file.is_open())
    {
        std::cerr << "Error opening file.\n";
        return 1;
    }

    nn::Mat t(9990, 69);

    std::string line;
    short index = 0;
    std::getline(file, line);

    for(size_t i = 0; i < 9990; ++i)
    {
        std::getline(file, line);
        std::vector<std::string> tokens = split(line, ','); 
        nn::Mat middle(1, 69);
        middle.fill(0);
        for (size_t j = 2; j < tokens.size()-1; ++j)
        {
            middle.setAt(0, j, std::stof(tokens[j]));
        }
        std::string s = tokens.back();
        s.erase(s.size() - 1);
        if(s != generas[index])
            ++index;

        middle.setAt(0, 59 + index, 1);
        t = t + middle;
    }

    t.print();
    file.close(); 
    std::ofstream out("csvMat.mat", std::ios::binary);

    t.save(out);
}

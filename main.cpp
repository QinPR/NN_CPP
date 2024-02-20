// main.cpp

// don't forget to include out neural network
#include <fstream> 
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>


#include "NeuralNetwork.hpp"


void ReadCSV(std::string filename, std::vector<RowVector*>& data)
{
	data.clear();
	std::ifstream file(filename);
	std::string line, word;
	// determine number of columns in file
	getline(file, line, '\n');
	std::stringstream ss(line);
	std::vector<Scalar> parsed_vec;
	while (getline(ss, word, ',')) {
		parsed_vec.push_back(Scalar(std::stof(&word[0]))); 
	}
	uint cols = parsed_vec.size();
	data.push_back(new RowVector(cols));
	for (uint i = 0; i < cols; i++) {
		data.back()->coeffRef(i) = parsed_vec[i];   // coeffRef: index the entry in RowVector
	}

	// read the file
	if (file.is_open()) {
		while (getline(file, line, '\n')) {    // get each sample (each sample is a line)
			std::stringstream ss(line);
			data.push_back(new RowVector(1, cols));
			uint i = 0;
			while (getline(ss, word, ',')) {	// get features in this sample 
				data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
				i++;
			}
		}
	}
}


//... data generator code here
void genData(std::string filename, uint input_feat_len)
{
	std::ofstream file1(filename + "-in");
	std::ofstream file2(filename + "-out");
	for (uint r = 0; r < 1000; r++) {
		// randomly generate samples with 12 input features
		Scalar y = 10;
		for (int i = 0; i < input_feat_len; i++) {
			Scalar feat_i = rand() / Scalar(RAND_MAX);
			if (i != input_feat_len - 1) {
				file1 << feat_i << ",";
			} else {
				file1 << feat_i << std::endl;
			}
			y += 8 * feat_i;
		}
		// suppose the ground truth distribution is y =  2 * x1 + 10 + x2
		file2 << y << std::endl;
	}
	file1.close();
	file2.close();
}


typedef std::vector<RowVector*> data;
int main()
{
	NeuralNetwork neural_nerwork({ 12, 128, 64, 1 });   // init neural network with specific topology 
	data dataset_X, dataset_Y;
	genData("test", 12);

	ReadCSV("test-in", dataset_X);
	ReadCSV("test-out", dataset_Y);
	neural_nerwork.train(dataset_X, dataset_Y);
	return 0;
}

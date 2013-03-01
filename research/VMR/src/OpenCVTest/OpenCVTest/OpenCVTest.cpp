// OpenCVTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cbir.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\flann\flann.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <string>
#include <map>

using namespace boost::filesystem;
using namespace std;
using namespace cv;

void readme();

/** @function main */
int main(int argc, char** argv) {
	if( argc < 3 )	{ readme(); return -1; }
	
	path p(argv[1]);
	std::string mode(argv[2]);

	if (!exists(p) || !is_directory(p)) { readme(); return -1; }
	if (kVocab == mode) {
		std::cout << "creating vocabulary.\n";
		std::cout << "img_directory : " << p << std::endl;
		cv::Mat vocabulary = extractTrainingVocabulary(p);
		p /= "surf_data.yml";
		std::cout << "writing vocabulary to : " << p << std::endl;
		writeMatToFile(p, vocabulary, kVocab);
	} else if (kSurfHist == mode) {
		std::cout << "generating surf histograms.\n";
		path filename("surf_data.yml");
		cv::Mat histograms = extractVocabHistograms(p, filename);
		std::cout << "writing histograms to file.\n";
		p /= "surf_hists.yml";
		writeMatToFile(p, histograms, kSurfHist);
	} else if (kIndex == mode) {
		std::cout << "generating search index.\n";
		cv::Mat histograms = readMatFromFile(p / "surf_hists.yml", kSurfHist);
		cv::flann::Index index = generateSearchIndex(histograms);
		std::cout << "saving index to : " << p << std::endl;
		index.save((p / "search.idx").string());
	} else if (kSearchIndex == mode) {
		std::cout << "search index for similar images.\n";
		if (argc != 4) { readme(); return -1; }
		path img_path(argv[3]);
		searchIndex(p, img_path);
	} else if (kSearchInvert == mode) {
		std::cout << "search inverted file list for similar images.\n";
		if (argc != 4) { readme(); return -1; }
		path img_path(argv[3]);
		searchInvert(p, img_path);
	} else if (kCountLines == mode) {
		std::cout << "go through all images and count the number of lines.\n";
		std::map<std::string, cv::Mat> line_descriptors;
		std::vector<std::string> class_names;
		countLines(p, line_descriptors, class_names);

		p /= "line_descriptors.yml";
		std::cout << "writing descriptors to : " << p << std::endl;
		cv::FileStorage fs(p.string(), cv::FileStorage::WRITE);
		fs << "classes" << class_names;
		
		for (std::map<std::string, cv::Mat>::const_iterator it = line_descriptors.begin();
			it != line_descriptors.end(); ++it) {
				fs << it->first << it->second;
		}
		fs.release();
	} else if (kBuildLineClassifier == mode) {
		std::cout << "build svm classifier on line descriptors.\n";
		std::map<std::string, cv::SVM> classifiers;
		buildClassifiers(p, classifiers);
	} else if (kColorHistograms == mode) {
		std::cout << "extract color histograms.\n";
		cv::Mat color_hists = extractColorHistograms(p);
		p /= "color_histograms.yml";
		writeMatToFile(p, color_hists, kColorHistograms);
	} else if (kSearchColor == mode) {
		std::cout << "search color histograms.\n";
		
		// get query image
		if (argc != 4) { readme(); return -1; }
		path img_path(argv[3]);
		vector<string> filenames;

		// search for the matching images
		searchColor(p, img_path, filenames);
	
		// show search results
		displayResults(img_path.string(), filenames);
	} else if (kSearchSURF == mode) {
		std::cout << "search SURF histograms.\n";
		
		// get query image
		if (argc != 4) { readme(); return -1; }
		path img_path(argv[3]);
		vector<string> filenames;

		// search for the matching images
		searchSURFHists(p, img_path, filenames);
	
		// show search results
		displayResults(img_path.string(), filenames);

	} else if (kSearchDecide == mode) {
		std::cout << "decide between SURF or color.\n";
		
		// get query image
		if (argc != 5) { readme(); return -1; }
		const float kThreshold = atof(argv[3]);
		path img_path(argv[4]);
		vector<string> filenames;

		// search for the matching images
		searchDecideSURFColor(p, img_path, kThreshold, filenames);
	
		// show search results
		displayResults(img_path.string(), filenames);

	}
	return 0;
}

/** @function readme */
void readme() {
	std::cout << " Usage: OpenCVTest.exe <img_directory> <command> [structure_threshold] [query_image_path]" << std::endl;
}


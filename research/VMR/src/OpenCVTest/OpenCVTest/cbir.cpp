#include "cbir.h"
#include <omp.h>
#include <string>
#include <exception>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/flann/flann.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/filesystem.hpp>
//#define _WIN32_WINNT 0x0502
#include <iostream>

using namespace boost::filesystem;
using namespace std;
using namespace cv;

cv::Mat readMatFromFile(const path kFilepath, const std::string kKey) {
	cv::FileStorage file_storage(kFilepath.string(), cv::FileStorage::READ);
	cv::Mat data;
	file_storage[kKey] >> data;
	file_storage.release();
	return data;
}

void writeMatToFile(const path kFilepath, const cv::Mat kData, const std::string kKey) {
	cv::FileStorage file_storage(kFilepath.string(), cv::FileStorage::WRITE);
	file_storage << kKey << kData;
	file_storage.release();
	return;
}

/*void writeClassifierToFile(const path kFilepath, const CvSVM kSVM, const std::string kKey) {
cv::FileStorage file_storage(kFilepath.string(), cv::FileStorage::WRITE);
file_storage << kKey << kSVM;
file_storage.release();
return;
}

CvSVM readClassifierFromFile(const path kFilepath, const std::string kKey) {
cv::FileStorage file_storage(kFilepath.string(), cv::FileStorage::READ);
CvSVM classifier;
file_storage[kKey] >> classifier;
file_storage.release();
return classifier;
}//*/

/* lists all filenames in dir */
void listDir(path dir, std::vector<path>& vec) {
	std::copy(directory_iterator(dir), directory_iterator(), std::back_inserter(vec));
	std::sort(vec.begin(), vec.end());
}

/* lists filenames of images in a directory */
void listSubDirectories(path dir, std::vector<path>& sub_directories) {
	/* get all filenames in directory */
	std::copy(directory_iterator(dir), directory_iterator(), std::back_inserter(sub_directories));
	std::sort(sub_directories.begin(), sub_directories.end());

	/* determine all non-sub-directories */
	std::vector<path> non_sub_directories;
	for (std::vector<path>::iterator it = sub_directories.begin(); it != sub_directories.end(); ++it) {
			const std::string kExt = (*it).extension().string();
			if (!is_directory(*it))
				non_sub_directories.push_back(*it);
	}

	/* remove all non-image filenames */
	for (std::vector<path>::const_iterator it = non_sub_directories.begin(); it != non_sub_directories.end(); ++it) {
			sub_directories.erase(
				std::remove(sub_directories.begin(), sub_directories.end(), *it), sub_directories.end());
	}
	//sub_directories.erase(std::remove(sub_directories.begin(), sub_directories.end(), dir / "index.idx"), sub_directories.end());
	//sub_directories.erase(std::remove(sub_directories.begin(), sub_directories.end(), dir / "surf_data.yml"), sub_directories.end());
}

/* lists all images that are present in all subdirectories */
void listSubDirImgs(path dir, std::vector<path>& vec) {
	vec.clear();
	// get all subdirectories
	vector<path> sub_directories;
	listSubDirectories(dir, sub_directories);

	// list all images from all sub_directories
	for (int i = 0; i < sub_directories.size(); ++i) {
		vector<path> images;
		const path kDir = sub_directories.at(i);
		listImgs(kDir, images);
		vec.insert(vec.end(), images.begin(), images.end());
	}

	// important to note that image names are sorted within a folder, but not across
	//  all images.
}

/* lists filenames of images in a directory */
void listImgs(path dir, std::vector<path>& vec) {
	/* get all filenames in directory */
	std::copy(directory_iterator(dir), directory_iterator(), std::back_inserter(vec));
	std::sort(vec.begin(), vec.end());

	/* determine all non-images */
	std::vector<path> non_images;
	for (std::vector<path>::iterator it = vec.begin(); it != vec.end(); ++it) {
			const std::string kExt = (*it).extension().string();
			if (kExt != ".png" && kExt != ".jpg")
				non_images.push_back(*it);
	}

	/* remove all non-image filenames */
	for (std::vector<path>::const_iterator it = non_images.begin(); it != non_images.end(); ++it) {
			vec.erase(std::remove(vec.begin(), vec.end(), *it), vec.end());
	}
	//vec.erase(std::remove(vec.begin(), vec.end(), dir / "index.idx"), vec.end());
	//vec.erase(std::remove(vec.begin(), vec.end(), dir / "surf_data.yml"), vec.end());
}

cv::Mat extractTrainingVocabulary(path training_dir) {
	/* get list of image names */
	std::vector<path> image_names;
	listSubDirImgs(training_dir, image_names);

	/* extract features from training images */
	cv::Ptr<cv::OpponentColorDescriptorExtractor> extractor(new cv::OpponentColorDescriptorExtractor(cv::Ptr<cv::SurfDescriptorExtractor>(new cv::SurfDescriptorExtractor())));
	cv::Mat training_descriptors(1, extractor->descriptorSize(), extractor->descriptorType());
	cv::Mat descriptors(1, extractor->descriptorSize(), extractor->descriptorType());
	std::vector<cv::KeyPoint> keypoints;
	//const int kHessianThreshold = 400;
	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector());

	std::cout << "processing : " << image_names.size() << " images" << std::endl;
	srand(time(NULL));
	int num_files = 0;
	for (std::vector<path>::const_iterator it(image_names.begin()); it != image_names.end(); ++it) {
		if ((++num_files % 200) == 0) 
			cout << endl << num_files;
		std::cout << ".";
		cv::Mat img = cv::imread(it->string());
		detector->detect(img, keypoints);
		extractor->compute(img, keypoints, descriptors);
		const int kStep = descriptors.rows / 10;  // only cluster 10% of descriptors to cut down on size
		const int kStartRow = (rand() % 10) * kStep;
		training_descriptors.push_back(descriptors.rowRange(kStartRow, kStartRow + kStep));
	}
	std::cout << std::endl;

	/* create vocabulary */
	const int kNumWords = 4000;

	cv::BOWKMeansTrainer bow_trainer(kNumWords);
	std::cout << "adding descriptors to trainer.\n";
	bow_trainer.add(training_descriptors);
	std::cout << "clustering descriptors into " << kNumWords << " clusters." << std::endl;
	cv::Mat vocabulary = bow_trainer.cluster(); 

	return vocabulary;
}


/** extract common colors from each image and then cluster them into a color code_book. */
cv::Mat extractLabVocabulary(path training_dir) {
	Mat histograms = Mat(0, 1, CV_32FC3);

	// get list of all class subfolders
	std::vector<path> sub_directories;
	listSubDirectories(training_dir, sub_directories);
	cout << " sub_directories : " << sub_directories.size() << endl;
	
	// go through each sub-folder, extracting histograms for all images present
	srand(time(NULL));
	for (int dir_idx = 0; dir_idx < sub_directories.size(); ++dir_idx) {
		vector<path> filenames;
		
		cout << sub_directories.at(dir_idx).filename() << endl;

		listImgs(sub_directories.at(dir_idx), filenames);
		
		cout << "extracting Lab colors from: " << filenames.size() << " images" << endl;
	
		//  loop over all files in current sub-folder
		for (vector<path>::const_iterator it = filenames.begin(); it != filenames.end(); ++it) {
			cout << ".";
			//  randomly skip a third of the images
			if ((rand() % 3) == 0) continue;

			Mat img = imread(it->string(), CV_LOAD_IMAGE_COLOR);
			if (!img.data) {	continue;	}
						
			Mat img_lab;
			cvtColor(img, img_lab, CV_BGR2Lab);
			Mat common_colors = extractCommonLabColors(img_lab);
			histograms.push_back(common_colors);
		}  // end file loop
		cout << endl;
	}  // end directory loop

	/* now we have the most common colors from each block of the image */
	/* create vocabulary by clustering common colors */
	const int kNumWords = 256;
	cv::BOWKMeansTrainer bow_trainer(kNumWords);
	std::cout << "adding descriptors to trainer.\n";
	bow_trainer.add(histograms);
	std::cout << "clustering descriptors into " << kNumWords << " clusters." << std::endl;
	cv::Mat vocabulary = bow_trainer.cluster(); 

	return vocabulary;
}


std::vector<std::vector<int>> createInvertedFileList(cv::Mat histograms) {
	std::vector<std::vector<int>> inv_list;
	const int kVocabSize = 1000;	
	inv_list.resize(kVocabSize);
	for (int img_idx = 0; img_idx < histograms.rows; ++img_idx) {
		for (int word_idx = 0; word_idx < kVocabSize; ++word_idx) {
			/* if image contains vocab word, add image to list */
			if (0.0 < histograms.row(img_idx).at<float>(word_idx)) {
				inv_list.at(word_idx).push_back(img_idx);
			}
		}
	}
	return inv_list;
}

void extractVocabHistograms(path img_dir, path vocab_file, std::vector<cv::Mat>& histograms) {
	/* read in vocabulary */
	path data_file = img_dir / vocab_file;
	std::cout << "reading vocabulary from " << data_file << std::endl;
	cv::Mat vocabulary;
	vocabulary = readMatFromFile(data_file, kVocab);

	/* extract codeword histograms using vocabulary */
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat response_hist;
	cv::Mat img;
	std::map<std::string, cv::Mat> classes_training_data;

	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector());
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L2));
	cv::Ptr<cv::OpponentColorDescriptorExtractor> extractor(new cv::OpponentColorDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>(new cv::SurfDescriptorExtractor())));
	cv::Ptr<cv::BOWImgDescriptorExtractor> bow_img_desc_extrct(new cv::BOWImgDescriptorExtractor(extractor, matcher));
	cout << "surf vocabulary type " << vocabulary.type() << endl;
	bow_img_desc_extrct->setVocabulary(vocabulary);

	// get list of all class subfolders
	std::vector<path> sub_directories;
	listSubDirectories(img_dir, sub_directories);

	// go through each sub-folder, extracting histograms for all images present
	for (int dir_idx = 0; dir_idx < sub_directories.size(); ++dir_idx) {
		histograms.push_back(cv::Mat());
		std::vector<path> filenames;
		std::vector<std::string> classes_names;
		listImgs(sub_directories.at(dir_idx), filenames);
		int total_samples = 0;
		std::cout << "extracting codewords from : " << filenames.size() << " images" << std::endl;
		//#pragma omp parallel for schedule(dynamic, 3)
		for (int i = 0; i < filenames.size(); ++i) {
			cout << ".";
			img = cv::imread(filenames[i].string());
			detector->detect(img, keypoints);
			bow_img_desc_extrct->compute(img, keypoints, response_hist);
			//#pragma omp critical
			{
				if (0 == keypoints.size()) 
					response_hist = Mat::zeros(1, 4000, CV_32F);
				histograms.at(dir_idx).push_back(response_hist);
			}
		}
		if (histograms.at(dir_idx).rows != filenames.size()) {
			cerr << "*******ERROR: extracted " << histograms.at(dir_idx).rows << " from " << filenames.size() << " images.\n";
			exit(EXIT_FAILURE);
		}
		cout << endl;
	}
}


/**  loop over all sub-directories and extract the CIE_L*a*b* histograms from each image */
void extractLabHistograms(path img_dir, path vocab_file, std::vector<cv::Mat>& histograms) {
	/* read in vocabulary */
	path data_file = img_dir / vocab_file;
	std::cout << "reading vocabulary from " << data_file << std::endl;
	cv::Mat vocabulary;
	vocabulary = readMatFromFile(data_file, kLabVocab);

	// get list of all class subfolders
	std::vector<path> sub_directories;
	listSubDirectories(img_dir, sub_directories);

	// go through each sub-folder, extracting histograms for all images present
	for (int dir_idx = 0; dir_idx < sub_directories.size(); ++dir_idx) {
		histograms.push_back(cv::Mat());
		std::vector<path> filenames;
		std::vector<std::string> classes_names;
		listImgs(sub_directories.at(dir_idx), filenames);
		int total_samples = 0;
		std::cout << "extracting codewords from : " << filenames.size() << " images" << std::endl;
		//#pragma omp parallel for schedule(dynamic, 3)
		for (int i = 0; i < filenames.size(); ++i) {
			cout << ".";
			
			//  read in image
			Mat img = cv::imread(filenames[i].string());
			
			//  convert image to CIE-L*a*b* color space
			Mat lab_img;
			cvtColor(img, lab_img, CV_BGR2Lab);
			
			Mat response_hist = extractLabHistogram(lab_img, vocabulary);

			//#pragma omp critical
			{
				histograms.at(dir_idx).push_back(response_hist);
			}
		}
		cout << endl;
	}
}

cv::Mat extractVocabHistogram(path img_path, path vocab_file) {
	/* read in vocabulary */
	path data_file = vocab_file;
	std::cout << "reading vocabulary from " << data_file << std::endl;
	cv::Mat vocabulary;
	vocabulary = readMatFromFile(data_file, kVocab);

	/* extract codeword histograms using vocabulary */
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat response_hist;
	cv::Mat index_descriptors;
	cv::Mat img;
	std::map<std::string, cv::Mat> classes_training_data;

	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector());
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L2));
	cv::Ptr<cv::OpponentColorDescriptorExtractor> extractor(new cv::OpponentColorDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>(new cv::SurfDescriptorExtractor())));
	cv::Ptr<cv::BOWImgDescriptorExtractor> bow_img_desc_extrct(new cv::BOWImgDescriptorExtractor(extractor, matcher));

	bow_img_desc_extrct->setVocabulary(vocabulary);

	std::vector<std::string> classes_names;
	std::cout << "extracting codewords from : " << img_path << std::endl;
	img = cv::imread(img_path.string());
	cout << "read in image\n";
	detector->detect(img, keypoints);
	cout << "detected keypoints in image\n";
	bow_img_desc_extrct->compute(img, keypoints, response_hist);
	cout << "created vocab histogram\n";
	index_descriptors.push_back(response_hist);
	return index_descriptors;
}

void searchColor(path index_dir, path query_img, vector<string> &results) {
	// load pre-computed histograms
	cout << "reading color histograms from sub-directories.\n";
	vector<path> sub_dirs;
	listSubDirectories(index_dir, sub_dirs);
	Mat hists(0, 24, CV_32F);
	for (int i = 0; i < sub_dirs.size(); ++i) {
		const path kCurrDir = sub_dirs.at(i);
		Mat tmp = readMatFromFile(kCurrDir / "color_histograms.yml", kColorHistograms);
		if (tmp.type() != CV_32F) {
			tmp.convertTo(tmp, CV_32F);
		}
		
		hists.push_back(tmp);
		cout << "hists.empty() : " << hists.empty() << endl;
		cout << "hists.rows = " << hists.rows << endl;
	}
	cv::Mat img = imread(query_img.string());

	cout << "extracting color histogram from query image.\n";
	// extract color histogram from query image
	cvtColor(img, img, CV_BGR2HSV);
	cv::Mat query_hist = extractHSVHistogram(img);
	if (query_hist.type() != CV_32F) {
			query_hist.convertTo(query_hist, CV_32F);
	}
	// match query histogram with database images
	cerr << "adding previously extracted descriptors to matcher.\n";
	
	BFMatcher matcher(NORM_L2);
	//matcher.add(hists);

	cerr << "training matcher on color histograms.\n";
	const int kK = 10;
	vector<vector<DMatch>> matches;
	cout << "matching color histograms.\n";
	cout << "query_hist.type() : " << query_hist.type() << ", hists.type() : " << hists.type() << endl;
	cout << "query_hist.cols : " << query_hist.cols << ", hists.cols : " << hists.cols << endl;
	matcher.knnMatch(query_hist, hists, matches, kK);
	
	cout << "getting matching image filenames.\n";
	// get results filenames
	vector<path> filenames;
	listSubDirImgs(index_dir, filenames);
	
	for (int i = 0; i < matches.size(); ++i) {
		vector<DMatch> &m = matches.at(i);
		for (int j = 0; j < m.size(); ++j) {
			DMatch dm = m.at(j);
			printf("dmatch[%d]: dist:%f queryidx:%d trainidx:%d imgidx:%d \n", j, dm.distance, dm.queryIdx, dm.trainIdx, dm.imgIdx);
			const int kImgIndex = m.at(j).trainIdx;
			results.push_back(filenames.at(kImgIndex).string());
		}
	}
}

void searchLab(path index_dir, path query_img, vector<string> &results) {
	// load pre-computed histograms
	cout << "reading CIE L*a*b* histograms from sub-directories.\n";
	vector<path> sub_dirs;
	listSubDirectories(index_dir, sub_dirs);
	Mat hists(0, 256, CV_32F);
	for (int i = 0; i < sub_dirs.size(); ++i) {
		const path kCurrDir = sub_dirs.at(i);
		Mat tmp = readMatFromFile(kCurrDir / "lab_hists.yml", kLabHist);
		if (tmp.type() != CV_32F) {
			tmp.convertTo(tmp, CV_32F);
		}
		
		hists.push_back(tmp);
	}

	//  read in color vocabulary
	Mat vocab = readMatFromFile(index_dir / "lab_data.yml", kLabVocab);

	//  read in query image
	cv::Mat img = imread(query_img.string(), CV_LOAD_IMAGE_COLOR);

	cout << "extracting Lab histogram from query image.\n";
	
	// extract Lab histogram from query image
	Mat lab_img;
	cvtColor(img, lab_img, CV_BGR2Lab);
	cv::Mat query_hist = extractLabHistogram(lab_img, vocab);
	if (query_hist.type() != CV_32F) {
			query_hist.convertTo(query_hist, CV_32F);
	}
	// match query histogram with database images
	cerr << "adding previously extracted descriptors to matcher.\n";
	
	BFMatcher matcher(NORM_L2);
	//matcher.add(hists);

	const int kK = 10;
	vector<vector<DMatch>> matches;
	cout << "matching color histograms.\n";
	matcher.knnMatch(query_hist, hists, matches, kK);
	
	cout << "getting matching image filenames.\n";
	// get results filenames
	vector<path> filenames;
	listSubDirImgs(index_dir, filenames);
	
	for (int i = 0; i < matches.size(); ++i) {
		vector<DMatch> &m = matches.at(i);
		for (int j = 0; j < m.size(); ++j) {
			DMatch dm = m.at(j);
			printf("dmatch[%d]: dist:%f queryidx:%d trainidx:%d imgidx:%d \n", j, dm.distance, dm.queryIdx, dm.trainIdx, dm.imgIdx);
			const int kImgIndex = dm.trainIdx;
			results.push_back(filenames.at(kImgIndex).string());
		}
	}
}


void searchSURFHists(path index_dir, path query_img, vector<string> &results) {
	// load pre-computed histograms
	cout << "reading SURF histograms from sub-directories.\n";
	vector<path> sub_dirs;
	listSubDirectories(index_dir, sub_dirs);
	Mat hists;
	for (int i = 0; i < sub_dirs.size(); ++i) {
		const path kCurrDir = sub_dirs.at(i);
		vector<path> filenames;
		listImgs(kCurrDir, filenames);
		Mat tmp = readMatFromFile(kCurrDir / "surf_hists.yml", kSurfHist);
		
		if (tmp.type() != CV_32F) {
			tmp.convertTo(tmp, CV_32F);
		}
		hists.push_back(tmp);
	}
	
	/* read in vocabulary */
	std::cerr << "reading vocabulary from file.\n";
	cv::Mat vocab = readMatFromFile(index_dir / "surf_data.yml", kVocab);
	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector());
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L2));
	cv::Ptr<cv::OpponentColorDescriptorExtractor> extractor(new cv::OpponentColorDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>(new cv::SurfDescriptorExtractor())));
	cv::Ptr<cv::BOWImgDescriptorExtractor> bow_img_desc_extrct(new cv::BOWImgDescriptorExtractor(extractor, matcher));

	bow_img_desc_extrct->setVocabulary(vocab);
	std::cout << "vocabulary has " << vocab.size.p[0] << " codewords.\n";

	/* extract words in query image */
	cerr << "extracting SURF histogram from query image.\n";
	std::vector<cv::KeyPoint> key_points;
	cv::Mat query_hist;
	cv::Mat img = cv::imread(query_img.string(), CV_LOAD_IMAGE_COLOR);
	std::cerr << "image size: " << img.size.p[0] << " X " << img.size.p[1] << std::endl;
	detector->detect(img, key_points);
	std::cerr << "extract histogram\n";
	bow_img_desc_extrct->compute(img, key_points, query_hist);

	// match query histogram with database images
	const int kK = 10;
	vector<vector<DMatch>> matches;
	cerr << "matching surf histograms.\n";
	matcher->knnMatch(query_hist, hists, matches, kK);
	
	cerr << "getting matching image filenames.\n";
	// get results filenames
	vector<path> filenames;
	listSubDirImgs(index_dir, filenames);
	cout << filenames.size() << " images" << endl;
	
	for (int i = 0; i < matches.size(); ++i) {
		vector<DMatch> &m = matches.at(i);
		for (int j = 0; j < m.size(); ++j) {
			DMatch dm = m.at(j);
			printf("dmatch[%d]: dist:%f queryidx:%d trainidx:%d imgidx:%d \n", j, dm.distance, dm.queryIdx, dm.trainIdx, dm.imgIdx);
			const int kImgIndex = dm.trainIdx;
			results.push_back(filenames.at(kImgIndex).string());
		}
	}
}


void searchDecideSURFColor(path index_dir, path query_img, const float kThreshold, std::vector<std::string> &results) {
	bool use_surf = false;
	
	// load image
	cv::Mat img = imread(query_img.string(), CV_LOAD_IMAGE_COLOR);

	// extract structure in image
	const float kStructure = calcStructure(img);

	// decide whether to use SURF or color search
	cout << "structure: " << kStructure << endl;
	cout << "threshold: " << kThreshold << endl;
	if (kThreshold < kStructure)
		use_surf = true;
	cout << "use_surf: " << use_surf << endl;
	// perform search
	if (use_surf)
		searchSURFHists(index_dir, query_img, results);
	else
		searchColor(index_dir, query_img, results);
}

/** wrapper class so that I can use Vec3f as key in map */
class Vec3fWrap {
public:
	Vec3fWrap(Vec3f v) : vec_(v){}

	bool operator<(const Vec3fWrap& v1) const { 
		return (vec_[0] < v1.vec()[0]) && (vec_[1] < v1.vec()[1]) && (vec_[2] < v1.vec()[2]);
	}

	Vec3f vec() const { return vec_; }
private:
	Vec3f vec_;
};

/** extracts the most common Lab colors from all sub-images */
Mat extractCommonLabColors(Mat img) {
	Mat current_colors = Mat(0, 1, CV_32FC3);
	resize(img, img, Size(256, 256));
	Mat new_img;
	img.convertTo(new_img, CV_32FC3);
	const int kStep = 16;
	
	//  loop over all sub-image regions
	for (int row = 0; row < new_img.rows; row += kStep) {
		for (int col = 0; col < new_img.cols; col += kStep) {
			Rect rect(row, col, kStep, kStep);
			const Mat kPatch = new_img(rect);
			Vec3f color = extractCommonLabColor(kPatch);
			current_colors.push_back(color);
		}  // end col loop
	}  // end row loop
	return current_colors;
}

/** extracts the most common color in the image */
Vec3f extractCommonLabColor(Mat img) {
	map<Vec3fWrap, int> colors;  // map with pixel color as key and count as value
	int max_count = 0;
	Vec3f common_color = img.at<Vec3f>(0, 0);
	for (int row = 0; row < img.rows; ++row) {
		for (int col = 0; col < img.cols; ++col) {
			Vec3fWrap Lab(img.at<Vec3f>(row, col));
			if (colors[Lab]++ > max_count) {
				common_color = Lab.vec();
				max_count = colors.at(Lab);
			}
		}
	}
	return common_color;
}

/** extracts the histogram of HSV color values in the input image */
Mat extractHSVHistogram(Mat img) {
		vector<Mat> hsv_planes;
		split(img, hsv_planes);

		// establish histogram parameters
		float hue_range[] = {0, 180};
		float sat_range[] = {0, 256};
		float val_range[] = {0, 256};
		const float* kHueHistRange = { hue_range };
		const float* kSatHistRange = { sat_range };
		const float* kValHistRange = { val_range };
		bool uniform = true;
		bool accumulate = true;
		int hue_hist_size = 18;
		int sat_hist_size = 3;
		int val_hist_size = 3;
	
		Mat h_hist, s_hist, v_hist;
		
		// extract histograms in h, s, and v
		calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &hue_hist_size, &kHueHistRange, uniform, accumulate);
		calcHist(&hsv_planes[1], 1, 0, Mat(), s_hist, 1, &sat_hist_size, &kSatHistRange, uniform, accumulate);
		calcHist(&hsv_planes[2], 1, 0, Mat(), v_hist, 1, &val_hist_size, &kValHistRange, uniform, accumulate);

		// normalize to [ 0, 1 ]
		normalize(h_hist, h_hist);
		normalize(s_hist, s_hist);
		normalize(v_hist, v_hist);
		
		// merge into one histogram and store in output vector
		Mat merged_hist(1, hue_hist_size + sat_hist_size + val_hist_size, CV_32F);
		{
			int i = 0;
			for (; i < 18; ++i)
				merged_hist.at<float>(i) = h_hist.at<float>(i);
			for (; i < hue_hist_size + sat_hist_size; ++i)
				merged_hist.at<float>(i) = s_hist.at<float>(i - hue_hist_size);
			for (; i < hue_hist_size + sat_hist_size + val_hist_size; ++i)
				merged_hist.at<float>(i) = v_hist.at<float>(i - hue_hist_size - sat_hist_size);
		}
		return merged_hist;
}

/** goes through every pixel in the image and calculates nearest-neighbor in the 
	color vocabulary, and then increments the bin for that color. Normalized histogram is returned. */
Mat extractLabHistogram(Mat img, Mat vocab) {
	Mat hist = Mat::zeros(1, vocab.rows, CV_32F);

	//  make sure that image is float 1-channel so we can do the matching on the vocabulary
	Mat img_fc3;
	if (img.type() != CV_32FC3)
		img.convertTo(img_fc3, CV_32FC3);
	else
		img_fc3 = img;
	//  create matrix of colors
	Mat original_colors(0, 3, CV_32F);
	for (int row = 0; row < img_fc3.rows; ++row) {
		for (int col = 0; col < img_fc3.cols; ++col) {
			Vec3f Lab = img_fc3.at<Vec3f>(row, col);
			Mat mat_lab(1, 3, CV_32F);
			mat_lab.at<float>(0) = Lab[0];
			mat_lab.at<float>(1) = Lab[1];
			mat_lab.at<float>(2) = Lab[2];
			original_colors.push_back(mat_lab);
			//original_colors.row(row * img_fc3.cols + col).push_back(Lab);
		}
	}

	//  match each pixel color with the closest color from vocabulary
	Ptr<BFMatcher> matcher(new BFMatcher(NORM_L2));
	vector<DMatch> matches;
	
	//  we have to add the vocabulary to a vector because the opencv api for descriptormatchers
	//  assumes that you will have a 
	vector<Mat> vocab_vec;
	vocab_vec.push_back(vocab);
	matcher->add(vocab_vec);
	matcher->match(original_colors, matches);

	//  go through matches and count occurences
	for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
		++(hist.at<float>(it->trainIdx));

	// normalize to [ 0, 1 ]
	normalize(hist, hist);
		
	return hist;
}

void extractColorHistograms(path training_dir, std::vector<cv::Mat>& histograms) {
	cout << "extractColorHistograms(" << training_dir << ", histograms<Mat>);\n";
	int hue_hist_size = 18;
	int sat_hist_size = 3;
	int val_hist_size = 3;


	// get list of all class subfolders
	std::vector<path> sub_directories;
	listSubDirectories(training_dir, sub_directories);
	cout << " sub_directories : " << sub_directories.size() << endl;
	// go through each sub-folder, extracting histograms for all images present
	for (int dir_idx = 0; dir_idx < sub_directories.size(); ++dir_idx) {
		Mat current_histograms = Mat(0, hue_hist_size + sat_hist_size + val_hist_size, CV_32FC1);
		vector<path> filenames;
		cerr << "listImgs(" << training_dir / sub_directories.at(dir_idx) << ", filenames);\n";
		listImgs(sub_directories.at(dir_idx), filenames);
		cerr << "extracting hsv histograms from: " << filenames.size() << " images" << endl;
	
		for (vector<path>::const_iterator it = filenames.begin(); it != filenames.end(); ++it) {
			Mat img = imread(it->string(), CV_LOAD_IMAGE_COLOR);
			if (!img.data) {	continue;	}

			// compute hsv histogram
			Mat img_hsv;
			cvtColor(img, img_hsv, CV_BGR2HSV);
			Mat hist = extractHSVHistogram(img_hsv);
			if (current_histograms.rows < 10) {
				cout << it->string() << ": " << hist << endl;
				waitKey();
			}
			current_histograms.push_back(hist);
		}
		histograms.push_back(current_histograms);
	}
}

cv::flann::Index generateSearchIndex(cv::Mat vocab_hist) {
	/* create search index for the descriptor histograms */
	std::cout << "creating search index.\n";
	const int kNumTrees = 5;
	cv::flann::KDTreeIndexParams index_params(kNumTrees);
	cv::flann::Index kdTree(vocab_hist, index_params);
	return kdTree;
}

std::string getClass(std::string filename) {
	const int kNumClasses = 7;
	const char *classes_init[] =  {"object", "rpg", "ak", "tank", "land", "heli", "unknown"};
	const vector<string> kClasses(classes_init, my_end(classes_init));
	for (int i = 0; i < kClasses.size(); ++i) {
		if (filename.find(kClasses[i]) != std::string::npos)
			return kClasses[i];
	}
	return kClasses[kClasses.size() - 1];
}

std::string getClass(const unsigned int kIndex) {
	const int kNumClasses = 7;
	const char *classes_init[] =  {"object", "rpg", "ak", "tank", "land", "heli", "unknown"};
	const vector<string> kClasses(classes_init, end(classes_init));
	return kClasses.at(kIndex);
}

void searchIndex(path index_dir, path query_filename) {
	const std::string kIndexName("search.idx");
	std::vector<path> filenames;
	listImgs(index_dir, filenames);

	std::cout << "reading histograms from file.\n";
	/* read in histograms */
	cv::Mat hists = readMatFromFile(index_dir / "surf_data.yml", kHist);
	std::cout << "num histograms : " << hists.size.p[0] << std::endl;
	cv::flann::SavedIndexParams index_params((index_dir / kIndexName).string());

	/* read in index */
	std::cout << "reading index from file.\n";
	cv::Mat index_mat(hists.size(), CV_32FC1);
	cv::flann::Index index(index_mat, index_params, cvflann::FLANN_DIST_EUCLIDEAN);

	/* read in vocabulary */
	std::cout << "reading vocabulary from file.\n";
	cv::Mat vocab = readMatFromFile(index_dir / "surf_data.yml", kVocab);
	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector());
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L2));
	cv::Ptr<cv::OpponentColorDescriptorExtractor> extractor(new cv::OpponentColorDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>(new cv::SurfDescriptorExtractor())));
	cv::Ptr<cv::BOWImgDescriptorExtractor> bow_img_desc_extrct(new cv::BOWImgDescriptorExtractor(extractor, matcher));

	bow_img_desc_extrct->setVocabulary(vocab);
	std::cout << "vocabulary has " << vocab.size.p[0] << " codewords.\n";

	/* extract words in query image */
	std::cout << "processing query image.\n";
	std::vector<cv::KeyPoint> key_points;
	cv::Mat query_hist;
	cv::Mat query_img = cv::imread(query_filename.string());
	std::cout << "image size: " << query_img.size.p[0] << " X " << query_img.size.p[1] << std::endl;
	detector->detect(query_img, key_points);
	bow_img_desc_extrct->compute(query_img, key_points, query_hist);

	/* search index for matches */
	std::vector<int> indices(1);
	std::vector<float> dists(1);
	const float *kQueryData = query_hist.ptr<float>(0);
	std::vector<float> query(kQueryData, kQueryData + query_hist.cols);
	std::cout << "query_hist size.p[0]:" << query_hist.size.p[0] << std::endl;
	std::cout << "searching index for features from query.\n";
	index.knnSearch(query, indices, dists, 10);  //, cv::flann::SearchParams(64));
	std::cout << "display index matches:\n";
	for (std::vector<path>::const_iterator it = filenames.begin(); 
		it != filenames.end();
		++it) 
		std::cout << *it << std::endl;

	std::vector<std::string> matches;
	for (int i = 0; i < indices.size(); ++i) {
		std::cout << "file, index, distance : " << filenames.at(indices.at(i)).string() << ", " << indices.at(i) << ", " << dists.at(i) << std::endl;
		matches.push_back(filenames.at(indices.at(i)).string());
	}
	displayResults(query_filename.string(), matches);

	/* use flann-based matcher */
	std::cout << "display flann matches:\n";
	cv::FlannBasedMatcher flann_matcher;
	std::vector<std::vector<cv::DMatch>> flann_matches;
	flann_matcher.knnMatch(query_hist, hists, flann_matches, 10);

	matches.clear();
	for (int i = 0; i < flann_matches.at(0).size(); ++i) {
		cv::DMatch dmatch = flann_matches.at(0).at(i);
		std::cout << "file, index, distance : " << filenames.at(dmatch.trainIdx).string() << ", " << dmatch.trainIdx << ", " << dmatch.distance << std::endl;
		matches.push_back(filenames.at(dmatch.trainIdx).string());
	}
	displayResults(query_filename.string(), matches);
}

class CompareCounts {
public:
	bool operator() (std::pair<int, int> p1, std::pair<int, int> p2) {
		if (p1.second < p2.second)
			return true;
		else
			return false;
	}
};

void searchInvert(path index_dir, path query_filename) {
	const std::string kIndexName("search.idx");
	std::vector<path> filenames;
	listImgs(index_dir, filenames);

	/* read in histograms */
	std::cout << "reading histograms from file.\n";
	cv::Mat hists = readMatFromFile(index_dir / "surf_data.yml", kHist);
	std::cout << "num histograms : " << hists.size.p[0] << std::endl;

	/* read in vocabulary */
	std::cout << "reading vocabulary from file.\n";
	cv::Mat vocab = readMatFromFile(index_dir / "surf_data.yml", kVocab);
	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector());
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L2));
	cv::Ptr<cv::OpponentColorDescriptorExtractor> extractor(new cv::OpponentColorDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>(new cv::SurfDescriptorExtractor())));
	cv::Ptr<cv::BOWImgDescriptorExtractor> bow_img_desc_extr(new cv::BOWImgDescriptorExtractor(extractor, matcher));
	bow_img_desc_extr->setVocabulary(vocab);

	std::cerr << "creating inverted file list.\n";
	std::vector<std::vector<int>> inv_list = createInvertedFileList(hists);

	/* extract descriptors in query image */
	std::cout << "processing query image.\n";
	std::vector<cv::KeyPoint> key_points;
	cv::Mat query_descriptors;
	cv::Mat query_img = cv::imread(query_filename.string());
	std::cout << "image size: " << query_img.size.p[0] << " X " << query_img.size.p[1] << std::endl;
	detector->detect(query_img, key_points);
	extractor->compute(query_img, key_points, query_descriptors);
	cv::Mat hist_matches;
	bow_img_desc_extr->compute(query_img, key_points, hist_matches);

	/* find vocab words */
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(query_descriptors, vocab, matches, 1);


	/* find all images with same descriptors */
	std::map<int, int> counter;
	for (int i = 0; i < matches.size(); ++i) {
		const int kWordIdx = matches.at(i).at(0).trainIdx;
		for (int j = 0; j < inv_list.at(kWordIdx).size(); ++j) {
			counter[inv_list.at(kWordIdx).at(j)]++;
		}
	}

	/* sort images based on most matches */
	std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, CompareCounts> p_queue;
	for (std::map<int, int>::const_iterator it = counter.begin();
		it != counter.end();
		++it) {
			p_queue.push(std::make_pair(it->first, it->second));
	}

	std::vector<std::string> result_names;
	for (int i = 0; i < std::min((int)p_queue.size(), 10); ++i) {
		std::pair<int, int> p = p_queue.top();
		p_queue.pop();
		std::cout << "file, index, num_matches : " << filenames.at(p.first).string() << ", " << p.first << ", " << p.second << std::endl;
		result_names.push_back(filenames.at(p.first).string());
	}
	displayResults(query_filename.string(), result_names);
}
const int kHistSize = 256;
const float kLog2 = 0.69314718056f;

float Log2(float n) {
	return (0.0f == n) ? 0.0f : log(n) / kLog2;
}

float calcEntropy(cv::Mat img) {
	//float entropy_accum = 0.0f;

	// separate each channel from image
	// std::vector<cv::Mat> bgr_channels;
	// cv::split(img, bgr_channels);

	// number of bins for each histogram
	const int kNumBins = 256;

	// set the ranges for each channel
	float range[] = {0, kNumBins};
	const float* kHistRange = {range};
	const bool kUniform = true;
	const bool kAccumulate = false;
	cv::Mat histogram;

	// calculate histograms
	calcHist(&img, 1, 0, cv::Mat(), histogram, 1, &kNumBins, &kHistRange,
		kUniform, kAccumulate);
	
	normalize(histogram, histogram, 1, 0, cv::NORM_L1);
	
	// calculate entropy
	//return computeShannonEntropy(histogram);
	return entropy(histogram);
}



float calcLineEntropy(const std::list<int> &kLineIndices, const std::vector<cv::Vec4i> &kLines) {
	const unsigned int kQuantizationFactor = 1;
	const unsigned int kNumBins = 180 / kQuantizationFactor;
	const cv::Vec4i kReferenceLine(0, 0, 1, 0);
	std::vector<float> angle_hist;
	angle_hist.resize(kNumBins);


	// create histogram of line angles
	for (std::list<int>::const_iterator it = kLineIndices.begin();
		it != kLineIndices.end(); ++it) {
			const cv::Vec4i kLine = kLines.at(*it);
			const float kAngle = getAngleBetweenLines(kReferenceLine, kLine);
			const unsigned int kAngleBin = static_cast<unsigned int>(kAngle / kQuantizationFactor);
			++(angle_hist.at(kAngleBin));
	}
	
	// normalize histogram
	const float kNormalFraction = 1.0f / static_cast<float>(kLineIndices.size());
	std::transform(angle_hist.begin(), angle_hist.end(), angle_hist.begin(),
		std::bind1st(std::multiplies<float>(), kNormalFraction));

	return entropy(angle_hist);
}

float getHistogramBinValue(cv::Mat hist, int binNum) {
	return hist.at<float>(binNum);
}

float getFrequencyOfBin(cv::Mat channel) {
	float frequency = 0.0;
	for(int i = 0; i < kHistSize; ++i) {
		float Hc = abs(getHistogramBinValue(channel, i));
		frequency += Hc;
	}
	return frequency;
}

float entropy(const std::vector<float> &kHist) {
	float entropy_accum = 0.0f;
	for (int i = 0; i < kHist.size(); ++i) {
		float val1 = kHist.at(i);
		float log_1 = Log2(val1);
		entropy_accum += val1 * log_1;
	}
	entropy_accum *= -1.0f;
	return (entropy_accum == entropy_accum) ? entropy_accum : 0.0f;
}

float entropy(const std::map<string, float> &kHist) {
	float entropy_accum = 0.0f;
	for (map<string, float>::const_iterator it = kHist.begin(); it != kHist.end(); ++it) {
		float val1 = it->second;
		float log_1 = Log2(val1);
		entropy_accum += val1 * log_1;
	}
	entropy_accum *= -1.0f;
	return (entropy_accum == entropy_accum) ? entropy_accum : 0.0f;
}

float entropy(const cv::Mat kHist) {
	float entropy_accum = 0.0f;
	for (int i = 0; i < kHist.cols; ++i) {
		float val1 = getHistogramBinValue(kHist, i);
		float log_1 = Log2(val1);
		entropy_accum += val1 * log_1;
	}
	return entropy_accum * -1.0f;
}

float computeShannonEntropy(const cv::Mat kHist) {
	float entropy = 0.0;
	float frequency = getFrequencyOfBin(kHist);
	for (int i = 0; i < kHistSize; ++i) {
		float Hc = abs(getHistogramBinValue(kHist,i));
		if (0 == Hc) continue;
		entropy += -(Hc / frequency) * log10((Hc / frequency));
	}
	entropy = entropy;
	//cout << entropy <<endl;
	return entropy;
}

void countLines(path img_dir, std::map<std::string, cv::Mat> &line_descriptors, std::vector<std::string> &class_names) {
	// get list of images
	std::vector<path> filenames;
	listImgs(img_dir, filenames);
	// extract line descriptors for each image
	for (std::vector<path>::const_iterator it = filenames.begin();
		it != filenames.end();
		++it) {
			cv::Mat color_image, image, edge_img, color_dest;
			
			// read image
			color_image = cv::imread(it->string());
			std::cout << it->string() << std::endl;
			cv::cvtColor(color_image, image, CV_RGB2GRAY);
			if (!(image.data)) {  // error reading image
				continue;
			}

			//float entropy = calcEntropy(image);
			//std::cout << "entropy: " << entropy << std::endl;

			const int kLowThreshold = 50;
			const int kHighThreshold = 200;
			const int kApertureSize = 3;
			//cv::medianBlur(edge_img, edge_img, 3);
			Canny(image, edge_img, kLowThreshold, kHighThreshold, kApertureSize);
			//cv::dilate(edge_img, edge_img, cv::Mat(), cv::Point(-1, -1));
			
			//cv::erode(edge_img, edge_img, cv::Mat(), cv::Point(-1, -1));
			cvtColor(edge_img, color_dest, CV_GRAY2BGR);

			// detect lines in the image
			std::vector<cv::Vec4i> lines;
			const double kDistanceResolution = 1;
			const double kAngleResolution = CV_PI / 180.0;
			const double kMinLineLength = 10.0;  //.05 * std::min<int>(image.cols, image.rows);
			const double kAccumThreshold = .9 * kMinLineLength;
			const double kMaxLineGap = .15 * kMinLineLength;
			cv::HoughLinesP(edge_img, lines, kDistanceResolution, kAngleResolution,
							kAccumThreshold, kMinLineLength, kMaxLineGap);
			//line_counts.push_back(lines.size());

			// remove lines that have high entropy
			cv::Mat entropy_map, entropy_img;
			getEntropyMap(image, lines, entropy_map);
			entropy_map.convertTo(entropy_img, CV_8UC1, 255.0 / 8.0);
			removeNoisyLines(edge_img, lines);

			// get parallel lines
			std::vector<cv::Vec4i> parallel_lines, parallel_groups, l_lines, u_lines;
			getParallelLines(lines, 5.0f, parallel_lines);
			getParallelGroups(parallel_lines, 5.0f, 0.9f, 2.0f, 0.5f, parallel_groups);
			const int kNumLines = lines.size();
			const int kNumParallelGroupsLines = parallel_groups.size();
			const float kDistThresh = 5.0f;
			const float kAngleThresh = 30.0f;
			getLJunctions(lines, kAngleThresh, kDistThresh, l_lines);
			getUJunctions(lines, kAngleThresh, kDistThresh, u_lines);
			const int kNumLLines = l_lines.size();
			const int kNumULines = u_lines.size();
			const float kRawX[3] = {static_cast<float>(kNumLLines) / kNumLines,
								static_cast<float>(kNumULines) / kNumLines,
								static_cast<float>(kNumParallelGroupsLines) / kNumLines};
			const cv::Mat kX(1, 3, CV_32FC1, (void*)kRawX);
			
			std::cout << " x: " << 1 / (.5 * ((1 / kX.at<float>(1)) + (1 / kX.at<float>(2)))) << std::endl;
			std::string class_ = getClass(it->string());
			if (line_descriptors.count(class_) == 0) {  // class hasn't been added yet
				line_descriptors[class_].create(0, kX.cols, kX.type());
				class_names.push_back(class_);
			}
			line_descriptors[class_].push_back(kX);
			// display parallel groups
			//color_dest = cv::Mat::zeros(color_dest.rows, color_dest.cols, color_dest.type());
			for (size_t i = 0; i < lines.size(); ++i) {
				cv::Vec4i line = lines.at(i);
				cv::line(color_dest, cv::Point(line[0], line[1]),
					cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255),
					1, CV_AA);
			}
			cv::imshow("original", color_image);
			cv::imshow("lines", color_dest);
			cv::imshow("entropy", entropy_img);

			cv::waitKey();
	}
}

float calcStructure(const cv::Mat kImg) {
	cv::Mat x;
	return extractLineDescriptor(kImg, x);
	cout << "x (outside) : " << x.at<float>(0) << " : " << x.at<float>(1) << " : " <<x.at<float>(2) << endl;
	
	vector<float> x_vec;
	x_vec.push_back(x.at<float>(1)); x_vec.push_back(x.at<float>(2));

//	float structure = calcHarmonicMean(x_vec);
	
	//std::cout << "manual x: " << 1 / (.5 * ((1 / x.at<float>(1)) + (1 / x.at<float>(2)))) << std::endl;
	//std::cout << "calculated x: " << structure << std::endl;

//	return structure;
}

float extractLineDescriptor(const cv::Mat kImg, cv::Mat &desc) {
	cv::Mat edge_img, color_dest;
	const int kLowThreshold = 50;
	const int kHighThreshold = 200;
	const int kApertureSize = 3;
	//cv::medianBlur(edge_img, edge_img, 3);
	Canny(kImg, edge_img, kLowThreshold, kHighThreshold, kApertureSize);
	//cv::dilate(edge_img, edge_img, cv::Mat(), cv::Point(-1, -1));

	//cv::erode(edge_img, edge_img, cv::Mat(), cv::Point(-1, -1));
	cvtColor(edge_img, color_dest, CV_GRAY2BGR);

	// detect lines in the image
	std::vector<cv::Vec4i> lines;
	const double kDistanceResolution = 1;
	const double kAngleResolution = CV_PI / 180.0;
	const double kMinLineLength = 10.0;  //.05 * std::min<int>(image.cols, image.rows);
	const double kAccumThreshold = .9 * kMinLineLength;
	const double kMaxLineGap = .15 * kMinLineLength;
	cv::HoughLinesP(edge_img, lines, kDistanceResolution, kAngleResolution,
		kAccumThreshold, kMinLineLength, kMaxLineGap);
	cout << "detected lines:" << lines.size() << endl;

	// remove lines that have high entropy
	//cv::Mat entropy_map, entropy_img;
	//getEntropyMap(kImg, lines, entropy_map);
	//entropy_map.convertTo(entropy_img, CV_8UC1, 255.0 / 8.0);
	//removeNoisyLines(edge_img, lines);

	// get parallel lines
	std::vector<cv::Vec4i> parallel_lines, parallel_groups, l_lines, u_lines;
	getParallelLines(lines, 5.0f, parallel_lines);
	cout << "detected parallel lines:" << parallel_lines.size() << endl;

	getParallelGroups(parallel_lines, 5.0f, 0.9f, 2.0f, 0.5f, parallel_groups);
	const int kNumLines = lines.size();
	const int kNumParallelGroupsLines = parallel_groups.size();
	const float kDistThresh = 5.0f;
	const float kAngleThresh = 30.0f;
	getLJunctions(lines, kAngleThresh, kDistThresh, l_lines);
	cout << "detected L lines:" << l_lines.size() << endl;
	getUJunctions(lines, kAngleThresh, kDistThresh, u_lines);
	cout << "detected U lines:" << u_lines.size() << endl;
	const int kNumLLines = l_lines.size();
	const int kNumULines = u_lines.size();
	const float kRawX[3] = {static_cast<float>(kNumLLines) / kNumLines,
		static_cast<float>(kNumULines) / kNumLines,
		static_cast<float>(kNumParallelGroupsLines) / kNumLines};
	desc = cv::Mat(1, 3, CV_32FC1, (void*)kRawX);
	cout << "desc (inside) : " << desc.at<float>(0) << " : " << desc.at<float>(1) << " : " << desc.at<float>(2) << endl;
	
	vector<float> x_vec;
	x_vec.push_back(desc.at<float>(1)); x_vec.push_back(desc.at<float>(2));

	return calcHarmonicMean(desc);
}

float sumInverse(float result, const float& kVal) {	return result + (1 / kVal);	}

float calcHarmonicMean(cv::Mat &data) {
	float result = 0.0f;
	result = 1 / (.5 * ((1 / data.at<float>(1)) + (1 / data.at<float>(2)))); 
	return result;
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(cv::Vec4i line_1, cv::Vec4i line_2, cv::Point2f &r) {
    cv::Point2i o1(line_1[0], line_1[1]);
	cv::Point2i p1(line_1[2], line_1[3]);
	cv::Point2i o2(line_2[0], line_2[1]);
	cv::Point2i p2(line_2[2], line_2[3]);
	cv::Point2i x = o2 - o1;
    cv::Point2i d1 = p1 - o1;
    cv::Point2i d2 = p2 - o2;

    float cross = d1.x * d2.y - d1.y * d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x) / cross;
    r = o1 + d1 * t1;
    return true;
}

void getEntropyMap(const cv::Mat &kImg, std::vector<cv::Vec4i> &lines, cv::Mat &entropy_map) {
	const unsigned int kWinScale = 8;
	const unsigned int kWinWidth = kImg.rows / kWinScale;
	const float kEntropyThreshold = 3.7;
	entropy_map = cv::Mat::zeros(kImg.rows, kImg.cols, CV_32FC1);
	std::cout << "img dim:0 size:" << kImg.size[0] << " dim:1 size:" << kImg.size[1] << std::endl;
	// calculate entropy of each window
	std::cout << "line angle entropy for each window\n";
	for (int row = 0; row < kImg.rows - kWinWidth; row += 2) {
		for (int col = 0; col < kImg.cols - kWinWidth; col += 2) {
			// find all lines that start in this window
			std::list<int> lines_in_window;
			const cv::Rect kWindow(row, col, kWinWidth, kWinWidth);
			for (int i = 0; i < lines.size(); ++i) {
				//if (kWindow.contains(cv::Point2i(lines.at(i)[0], lines.at(i)[1])))
				const cv::Vec4i kLine = lines.at(i);
				if (cv::clipLine(kWindow, cv::Point(kLine[0], kLine[1]), cv::Point(kLine[1], kLine[2]))) {
					lines_in_window.push_back(i);
				}
			}

			// compute entropy of line angles in this window
			const float kEntropy = calcLineEntropy(lines_in_window, lines);
			const unsigned int kRectCenterRow = row + .5 * kWinWidth;
			const unsigned int kRectCenterCol = col + .5 * kWinWidth;
			entropy_map.at<float>(kRectCenterRow, kRectCenterCol) = kEntropy;			
		}
	}
}

// looks at the angle of all of the lines in an image and removes the lines 
// in regions where the entropy of the line directions is high
void removeNoisyLines(const cv::Mat &kImg, std::vector<cv::Vec4i> &lines) {
	const int kNumBoxes = 4;
	const int kColStep = kImg.cols / kNumBoxes;
	const int kRowStep = kImg.rows / kNumBoxes;
	const float kEntropyThreshold = 3.6;
	
	// initialize vectors
	std::vector<std::list<int>> lines_in_regions;
	const int kNumRegions = kNumBoxes * kNumBoxes;
	lines_in_regions.resize(kNumRegions);
	for (int i = 0; i < kNumRegions; ++i)
		lines_in_regions.at(i).clear();

	// place lines in quadrants
	for (int i = 0; i < lines.size(); ++i) {
		const int kBlockCol = std::min(lines.at(i)[0] / kColStep, kNumBoxes - 1);
		const int kBlockRow = std::min(lines.at(i)[1] / kRowStep, kNumBoxes - 1);
		const int kBlockIndex = kBlockRow * kNumBoxes + kBlockCol;
		lines_in_regions.at(kBlockIndex).push_back(i);
	}

	// calculate entropy of each line quadrant
	std::cout << "line angle entropy for each quadrant\n";
	std::map<int, bool> lines_to_exclude;
	for (std::vector<std::list<int>>::iterator it = lines_in_regions.begin();
		it != lines_in_regions.end(); ++it) {
			const float kRegionEntropy = calcLineEntropy(*it, lines);
			std::cout << "  " << kRegionEntropy << std::endl;
			if (kRegionEntropy > kEntropyThreshold) {
				for (std::list<int>::const_iterator it2 = it->begin();
					it2 != it->end(); ++it2) {
						lines_to_exclude[*it2] = true;
				}
			}
	}

	// remove lines back-to-front so that remaining indices stay correct.
	std::vector<cv::Vec4i> new_lines;
	new_lines.reserve(lines.size());
	for (int i = 0; i < lines.size(); ++i) {
		if (lines_to_exclude[i] == false)
			new_lines.push_back(lines.at(i));
	}

	// move updated lines to return vector
	lines.swap(new_lines);
}


void getLongerLines(const std::vector<cv::Vec4i> &kLines, const float kMinLength,
	std::vector<cv::Vec4i> &long_lines) {
		for (std::vector<cv::Vec4i>::const_iterator it = kLines.begin();
			it != kLines.end(); ++it) {
				if (kMinLength <= getDistance(*it)) // is line long enough?
					long_lines.push_back(*it);
		}
}

void getCoterminations(const std::vector<cv::Vec4i> &kLines, const float kSimilarityAngle,
	const float kDistanceThreshold, std::vector<cv::Vec4i> &coterm_lines) {
		std::map<int, int> coterm_lookup;
		// examine all line pairs
		for (int i = 0; i < kLines.size(); ++i) {
			const cv::Vec4i kLine1 = kLines.at(i);
			for (int j = 1; j < kLines.size(); ++j) {
				const cv::Vec4i kLine2 = kLines.at(j);
				const float kAngle = getAngleBetweenLines(kLine1, kLine2);
				if (kSimilarityAngle <= kAngle &&
					((180 - kSimilarityAngle) >= kAngle)) {
						const float kDiffX = getDiffX(kLine1, kLine2);
						const float kDiffY = getDiffY(kLine1, kLine2);
						if (kDistanceThreshold >= std::max<float>(kDiffX, kDiffY)) {
							// lines are coterminant
							coterm_lookup[i] = i;
							coterm_lookup[j] = j;
						}
				}
			}
		}

		// place coterminant lines in output vector
		for (std::map<int, int>::const_iterator it = coterm_lookup.begin();
			it != coterm_lookup.end();
			++it) {
				coterm_lines.push_back(kLines.at(it->first));
		}
}

void getLJunctions(const std::vector<cv::Vec4i> &kLines, const float kDeltaLAngle,
	const float kDistanceThreshold, std::vector<cv::Vec4i> &l_junct_lines) {
		std::map<int, int> l_junct_lookup;
		// examine all line pairs
		for (int i = 0; i < kLines.size(); ++i) {
			const cv::Vec4i kLine1 = kLines.at(i);
			for (int j = i + 1; j < kLines.size(); ++j) {
				const cv::Vec4i kLine2 = kLines.at(j);
				const float kAngle = getAngleBetweenLines(kLine1, kLine2);
				if (kDeltaLAngle > abs(90 - kAngle)) {
						const float kDiffX = getDiffX(kLine1, kLine2);
						const float kDiffY = getDiffY(kLine1, kLine2);
						if (kDistanceThreshold >= std::max<float>(kDiffX, kDiffY)) {
							// lines are part of an L junction
							l_junct_lookup[i] = i;
							l_junct_lookup[j] = j;
						}
				}
			}
		}

		// place L lines in output vector
		for (std::map<int, int>::const_iterator it = l_junct_lookup.begin();
			it != l_junct_lookup.end();
			++it) {
				l_junct_lines.push_back(kLines.at(it->first));
		}
}

void getUJunctions(const std::vector<cv::Vec4i> &kLines, const float kDeltaLAngle,
	const float kDistanceThreshold, std::vector<cv::Vec4i> &u_junct_lines) {
		std::map<int, std::vector<int>> l_junct_lookup;
		
		// examine all line pairs for l junctions
		for (int i = 0; i < kLines.size(); ++i) {
			const cv::Vec4i kLine1 = kLines.at(i);
			for (int j = i + 1; j < kLines.size(); ++j) {
				const cv::Vec4i kLine2 = kLines.at(j);
				const float kAngle = getAngleBetweenLines(kLine1, kLine2);
				if (kDeltaLAngle > abs(90 - kAngle)) {
						const float kDiffX = getDiffX(kLine1, kLine2);
						const float kDiffY = getDiffY(kLine1, kLine2);
						if (kDistanceThreshold >= std::max<float>(kDiffX, kDiffY)) {
							// lines are part of an L junction
							l_junct_lookup[i].push_back(j);
							l_junct_lookup[j].push_back(i);
						}
				}
			}
		}
		
		std::map<int, int> u_junct_lookup;
		// go through all l junction vectors
		for (std::map<int, std::vector<int>>::const_iterator map_it = l_junct_lookup.begin();
			map_it != l_junct_lookup.end();	++map_it) {
				// go through all lines in l-junctions with current line
				const cv::Vec4i kSharedLine = kLines.at(map_it->first);
				for (std::vector<int>::const_iterator vec_it_1 = map_it->second.begin();
					vec_it_1 != map_it->second.end() - 1;	++vec_it_1) {
						// get line from first l-junction
						const cv::Vec4i kLine1 = kLines.at(*vec_it_1);
						
						// are both points in line 1 on same side of shared line
						cv::Vec2i point;
						point[0] = kLine1[0];  point[1] = kLine1[1];
						const bool kLine1Point0IsLeft = isLeftOfLine(kSharedLine, point);
						point[0] = kLine1[2];  point[1] = kLine1[3];
						const bool kLine1Point1IsLeft = isLeftOfLine(kSharedLine, point);
						
						// if the line isn't all on the same side then don't include it
						if (kLine1Point0IsLeft != kLine1Point1IsLeft) continue;

						for (std::vector<int>::const_iterator vec_it_2 = vec_it_1 + 1;
							vec_it_2 != map_it->second.end(); ++vec_it_2) {
								const cv::Vec4i kLine2 = kLines.at(*vec_it_2);
								point[0] = kLine2[0];  point[1] = kLine2[1];
								const bool kLine2Point0IsLeft = isLeftOfLine(kSharedLine, point);
								point[0] = kLine2[2];  point[1] = kLine2[3];
								const bool kLine2Point1IsLeft = isLeftOfLine(kSharedLine, point);
						
								// if the line isn't all on the same side then don't include it,
								//   or if line 2 isn't on the same side of the shared line as
								//   line 1, then don't include it
								if ((kLine2Point0IsLeft != kLine2Point1IsLeft) ||
									(kLine2Point0IsLeft != kLine1Point0IsLeft)) continue;

								// both lines are on the same side of the shared line
								u_junct_lookup[map_it->first] = map_it->first;
								u_junct_lookup[*vec_it_1] = *vec_it_1;
								u_junct_lookup[*vec_it_2] = *vec_it_2;
						}
				}
		}

		// place L lines in output vector
		for (std::map<int, int>::const_iterator it = u_junct_lookup.begin();
			it != u_junct_lookup.end();
			++it) {
				u_junct_lines.push_back(kLines.at(it->first));
		}
}

bool isLeftOfLine(const cv::Vec4i kLine, cv::Vec2i kPoint) {
	return ((kLine[2] - kLine[0]) * (kPoint[1] - kLine[1]) -
		(kLine[3] - kLine[1]) * (kPoint[0] - kLine[0])) > 0;
}

float getDistance(const cv::Vec4i kLine) {
	const float kXDiff = static_cast<float>(kLine[0] - kLine[2]);
	const float kYDiff = static_cast<float>(kLine[1] - kLine[3]);
	return sqrt(kXDiff * kXDiff + kYDiff * kYDiff);
}

float getDiffY(const cv::Vec4i kLine1, const cv::Vec4i kLine2) {
	float diff = 0.0f;
	diff = std::min<float>(abs(kLine1[1] - kLine2[1]), abs(kLine1[1] - kLine2[3]));
	diff = std::min<float>(diff, abs(kLine1[3] - kLine2[1]));
	diff = std::min<float>(diff, abs(kLine1[3] - kLine2[3]));
	return diff;
}

float getDiffX(const cv::Vec4i kLine1, const cv::Vec4i kLine2) {
	float diff = 0.0f;
	diff = std::min<float>(abs(kLine1[0] - kLine2[0]), abs(kLine1[0] - kLine2[2]));
	diff = std::min<float>(diff, abs(kLine1[2] - kLine2[0]));
	diff = std::min<float>(diff, abs(kLine1[2] - kLine2[2]));
	return diff;
}

// returns angle, in degrees, between two lines
float getAngleBetweenLines(const cv::Vec4i kLine1, const cv::Vec4i kLine2) {
	const float kYDiff1 = static_cast<float>(kLine1[3] - kLine1[1]);
	const float kYDiff2 = static_cast<float>(kLine2[3] - kLine2[1]);

	const float kXDiff1 = static_cast<float>(kLine1[2] - kLine1[0]);
	const float kXDiff2 = static_cast<float>(kLine2[2] - kLine2[0]);
	
	const float kAngle1 = atan2(kYDiff1, kXDiff1) * 180.0f / CV_PI;
	const float kAngle2 = atan2(kYDiff2, kXDiff2) * 180.0f / CV_PI;
	//std::cout << "angle1 : " << kAngle1 << ", angle2 : " << kAngle2 << std::endl;
	float angle_between = abs(kAngle1 - kAngle2);

	if (180 < angle_between)
		angle_between -= 180;
	
	return angle_between;
}

void getParallelLines(const std::vector<cv::Vec4i> &kLines, const float kSimilarityAngle,
	std::vector<cv::Vec4i> &parallel_lines) {
		std::map<int, int> parallel_lookup;
		// examine all line pairs
		for (int i = 0; i < kLines.size(); ++i) {
			const cv::Vec4i kLine1 = kLines.at(i);
			for (int j = 0; j < kLines.size(); ++j) {
				if (j == i) continue;

				const cv::Vec4i kLine2 = kLines.at(j);
				const float kAngle = getAngleBetweenLines(kLine1, kLine2);
				if (kSimilarityAngle >= kAngle || kSimilarityAngle >= (180 - kAngle)) {
					// lines are parallel
					parallel_lookup[i] = i;
					parallel_lookup[j] = j;
				}
			}
		}

		// place parallel lines in output vector
		for (std::map<int, int>::const_iterator it = parallel_lookup.begin();
			it != parallel_lookup.end();
			++it) {
				parallel_lines.push_back(kLines.at(it->first));
		}
}

void getParallelGroups(const std::vector<cv::Vec4i> &kParallelLines, const float kSimilarityAngle,
	const float kLengthRatio, const float kDistanceThreshold, const float kOverlapThreshold,
	std::vector<cv::Vec4i> &parallel_groups) {
		std::map<int, int> parallel_lookup;
		// examine all line pairs
		for (int i = 0; i < kParallelLines.size(); ++i) {
			for (int j = 0; j < kParallelLines.size(); ++j) {
				if (j == i) continue;
				cv::Vec4i line_1 = kParallelLines.at(i);
				cv::Vec4i line_2 = kParallelLines.at(j);
				
				// check for similar lengths
				float length_1 = getDistance(line_1);
				float length_2 = getDistance(line_2);
				if (length_1 > length_2) { // length_1 has to be smaller or equal to length_2
					cv::Vec4i tmp_vec = line_1;
					line_1 = line_2;
					line_1 = tmp_vec;
					
					float tmp = length_1;
					length_1 = length_2;
					length_2 = tmp;
				}
				if (length_1 / length_2 <= kLengthRatio) continue; // length not similar

				// lines must be "relatively" close
				if (getDistanceRatio(line_1, line_2) >= kDistanceThreshold) continue; // too far

				// sufficient overlap
				if (getOverlapRatio(line_1, line_2) <= kOverlapThreshold) continue; // not enough

				parallel_lookup[i] = i;
				parallel_lookup[j] = j;
			}
		}

		// place parallel lines in output vector
		for (std::map<int, int>::const_iterator it = parallel_lookup.begin();
			it != parallel_lookup.end();
			++it) {
				parallel_groups.push_back(kParallelLines.at(it->first));
		}
}

float getDistanceRatio(const cv::Vec4i kLine1, const cv::Vec4i kLine2) {
	cv::Vec2i mid_1 = getMidPoint(kLine1);
	cv::Vec2i mid_2 = getMidPoint(kLine2);
	cv::Vec4i mid_point_vector(mid_1[0], mid_1[1], mid_2[0], mid_2[1]);
	const float kMidPointDist = getDistance(mid_point_vector);
	const float kAvgLength = (getDistance(kLine1) + getDistance(kLine2)) / 2.0f;
	return kMidPointDist / kAvgLength;
}

float getOverlapRatio(const cv::Vec4i kLine1, const cv::Vec4i kLine2) {
	float overlap = 0.0f;
	
	// Y-axis projection overlap case
	cv::Vec4i y_proj(kLine1[1], kLine1[3], kLine2[1], kLine2[3]);
	
	// make sure first value is smaller than second
	if (y_proj[0] > y_proj[1]) 
		swap(y_proj[0], y_proj[1]);
	if (y_proj[2] > y_proj[3]) 
		swap(y_proj[2], y_proj[3]);

	// check overlap
	if ((y_proj[1] > y_proj[2]) && (y_proj[3] > y_proj[0])) {
		float overlap_distance = static_cast<float>(std::min<int>(y_proj[1], y_proj[3]) -
									std::max<int>(y_proj[0], y_proj[2]));
		overlap = overlap_distance / (y_proj[1] - y_proj[0]);
	}
	
	// X-axis projection overlap case
	cv::Vec4i x_proj(kLine1[0], kLine1[2], kLine2[0], kLine2[2]);
	
	// make sure first value is smaller than second
	if (x_proj[0] > x_proj[1]) 
		swap(x_proj[0], x_proj[1]);
	if (x_proj[2] > x_proj[3]) 
		swap(x_proj[2], x_proj[3]);

	// check overlap
	if ((x_proj[1] > x_proj[2]) && (x_proj[3] > x_proj[0])) {
		
		float overlap_distance = static_cast<float>(std::min<int>(x_proj[1], x_proj[3]) -
									std::max<int>(x_proj[0], x_proj[2]));
		overlap = std::max<float>(overlap, overlap_distance / (x_proj[1] - x_proj[0]));
	}

	// check projection overlap
	//cv::Vec4i v1(10, 10, 20, 20);
	//cv::Vec4i v2(10, 10, 30, 10);
	//float angle = getAngleBetweenLines(v1, v2);
	//float dist = getDistance(v1);
	//std::cout << "dist,angle between v1 and v2 : " << dist << ", " << angle << std::endl;
	//double proj_overlap = getDistance(v1) * cos(getAngleBetweenLines(v1, v2));
	//std::cout << "v1 dot v2 : " << proj_overlap << std::endl;
	return overlap;
}

cv::Vec2i getMidPoint(const cv::Vec4i kLine) {
	cv::Vec2i mid_point;
	mid_point[0] = abs(kLine[0] - kLine[2]);
	mid_point[1] = abs(kLine[1] - kLine[3]);
	return mid_point;
}

void displayResults(string query_filename, std::vector<std::string> &filenames) {
	const std::string kQuery("Query Image");
	const std::string kMatches("Matches");

	// first file name is the query image
	cv::namedWindow(kQuery);
	cv::imshow(kQuery, cv::imread(query_filename));

	// show matches
	cv::namedWindow(kMatches);
	for (int i = 0; i < filenames.size(); ++i) {
		cv::imshow(kMatches, cv::imread(filenames.at(i)));
		cv::waitKey();
	}

	// cleanup
	cv::destroyAllWindows();
}


void buildClassifiers(path p, std::map<std::string, cv::SVM> &classifiers) {
	path p2(p);
	p /= "line_descriptors.yml";
	std::cout << "reading line descriptors from " << p << std::endl;
	
	// read class names
	std::cout << "reading class names.\n";
	cv::FileStorage fs(p.string(), cv::FileStorage::READ);
	std::vector<std::string> class_names;
	fs["classes"] >> class_names;
	
	// read descriptors
	std::cout << "reading line descriptors for each class.\n";
	std::map<std::string, cv::Mat> line_descriptors;
	for (int i = 0; i < class_names.size(); ++i) {
		const std::string kClass = class_names.at(i);
		std::cout << "reading for " << kClass << std::endl;
		fs[kClass] >> line_descriptors[kClass];
	}
	fs.release();

	// train classifiers
	const float kTrainingSplit = 0.7f;
	std::cout << "creating classifiers for each class\n";
	for (int i = 0; i < class_names.size(); ++i) {
		const std::string kClass = class_names.at(i);
		std::cout << "  training for " << kClass << std::endl;
		cv::Mat training_data(0, line_descriptors.begin()->second.cols, line_descriptors.begin()->second.type());
		cv::Mat svm_responses(0, 1, CV_32FC1);

		// put positive examples in training data
		std::cout << "    positive examples\n";
		int end_training_row = kTrainingSplit * line_descriptors[kClass].rows;
		training_data.push_back(line_descriptors[kClass].rowRange(0, end_training_row));
		cv::Mat class_responses = cv::Mat::ones(end_training_row, 1, CV_32FC1);
		std::cerr << "      positive response dims = " << end_training_row << ", " << line_descriptors[kClass].cols << std::endl;
		svm_responses.push_back(class_responses);

		// put all other classes as negative training data
		std::cout << "    negative examples\n";
		for (std::map<std::string, cv::Mat>::const_iterator it = line_descriptors.begin();
			it != line_descriptors.end(); ++it) {
				if (it->first.compare(kClass) == 0) continue;
				std::cerr << "      " << it->first << std::endl;
				end_training_row = kTrainingSplit * it->second.rows;
				training_data.push_back(it->second.rowRange(0, end_training_row));
				std::cerr << "      pushed Mat data\n";
				std::cerr << "      negative response dims = " << end_training_row << ", " << it->second.cols << std::endl;
				class_responses = cv::Mat::zeros(end_training_row, 1, CV_32FC1);
				svm_responses.push_back(class_responses);
				std::cerr << "      pushed Mat responses\n";
		}
	
		// train the classifier for current class
		std::cerr << "    training...\n";
		cv::Mat training_data_32f;
		training_data.convertTo(training_data_32f, CV_32F);  // don't know why, but have to convert to 32F to train

		classifiers[kClass].train(training_data_32f, svm_responses);
		std::string svm_file = (p2 / std::string("svm_").append(kClass)).string();
		std::cerr << "saving svm to : " << svm_file << std::endl;
		classifiers[kClass].save(svm_file.c_str());
		//classifiers.push_back(svm);
		//std::cerr << "pushed classifier onto vector.\n";
	}

	// test each classifier

	// initialize confusion matrix
	std::map<std::string, std::map<std::string, int>> confusion_matrix;
	for (std::map<std::string, cv::SVM>::const_iterator it1 = classifiers.begin();
		it1 != classifiers.end(); ++it1) {
			for (std::map<std::string, cv::SVM>::const_iterator it2 = classifiers.begin();
				it2 != classifiers.end(); ++it2) {
					confusion_matrix[it1->first][it2->first] = 0;
			}
	}

	// test classifiers on training data
	for (std::map<std::string, cv::Mat>::const_iterator it = line_descriptors.begin();
		it != line_descriptors.end(); ++it) {
			const std::string kDataClass = it->first;
			for (int i = kTrainingSplit * it->second.rows; i < it->second.rows; ++i) {
				const cv::Mat kDescriptor = it->second.row(i);
				// check how each classifier classifies this descriptor
				for (std::map<std::string, cv::SVM>::const_iterator svm_it = classifiers.begin();
					svm_it != classifiers.end(); ++svm_it) {
						const std::string kClassifierClass = svm_it->first;
						const float kResponse = svm_it->second.predict(kDescriptor);
						if (1.0f == kResponse)
							confusion_matrix[kClassifierClass][kDataClass]++;
				}
			}
	}

	// output confusion matrix
	for (std::map<std::string, std::map<std::string, int>>::const_iterator class_it = confusion_matrix.begin();
		class_it != confusion_matrix.end(); ++class_it) {
			const std::string kClassString = class_it->first;
			std::cout << "\t" << kClassString << "\t";
			for (std::map<std::string, int>::const_iterator data_it = class_it->second.begin();
				data_it != class_it->second.end(); ++data_it) {
					const std::string kDataString = data_it->first;
					std::cout << kDataString << ":" << data_it->second << "\t";
			
			}
			std::cout << std::endl;
	}
}

void calcHistGain(vector<path>& filenames, Mat& hists, vector<float>& gain) {
	gain.clear();
	gain.resize(hists.cols);
	map<int, map<string, float>> word_to_class;
	const int kHistSize = hists.cols;
	map<string, float> class_probability;
	const int kNumClasses = 7;  // TODO : This is temporary just to get initial results
	
	// initialize words data structure
	cout << "initialize words data structure\n";
	for (int i = 0; i < hists.cols; ++i) {
		for (int j = 0; j < kNumClasses; ++j) {
			word_to_class[i][getClass(j)] = 0.0f;
		}
	}

	// go through each file and keep track of what words are present
	//  also, calculate global entropy
	cout << "calculate word counts in images and global entropy.\n";
	for (int i = 0; i < filenames.size(); ++i) {
		string class_ = getClass(filenames.at(i).filename().string());
		class_probability[class_]++;
		cout << filenames.at(i) << " : " << class_ << endl;
		for (int word_idx = 0; word_idx < kHistSize; ++word_idx) {
			if (0.0 < hists.row(i).at<float>(word_idx))  // if words are present
				word_to_class[word_idx][class_]++;
		}
	}

	// normalize class file counts
	cout << "normalize class file counts.\n";
	for (map<string, float>::iterator it = class_probability.begin(); it != class_probability.end(); ++it) {
			it->second /= filenames.size();
	}

	// place probabilities in vector for entropy calculation
	vector<float> class_probs_vec;
	for (unsigned int i = 0; i < kNumClasses; ++i)
		class_probs_vec.push_back(class_probability[getClass(i)]);

	// calculate the entropy of the entire training set
	const float kOriginalEntropy = entropy(class_probs_vec);
	cout << "entropy of whole set is : " << kOriginalEntropy << endl;

	// calculate the gain of each word
	cout << "calculate gain for each word";
	for (map<int, map<string, float>>::iterator word_it = word_to_class.begin(); word_it != word_to_class.end(); ++word_it) {
		cout << " " << word_it->first << " ";
		class_probs_vec.clear();
		const int kCurrentWord = word_it->first;
		float class_count = 0.0f;
		for (map<string, float>::iterator class_it = word_it->second.begin(); class_it != word_it->second.end(); ++class_it) {
			class_count += class_it->second;
		}

		// normalize class occurence counter for images that contain current word
		// This gives us the posterior probability for each class, given the current word is detected.
		if (0.0 < class_count) {
			for (map<string, float>::iterator class_it = word_it->second.begin(); class_it != word_it->second.end(); ++class_it) {
				class_it->second /= class_count;
			}

			// calculate entropy of data set where word is present
			const float kWordEntropy = entropy(word_it->second);

			// calculate reduction in entropy when word is present and store as information gain
			gain.at(word_it->first) = kOriginalEntropy - kWordEntropy;
		}
	}	
	cout << endl;
}

void calcHistGainSubdirectories(/*vector<path>& sub_directories, */map<string, cv::Mat>& hists, vector<float>& gain) {
	gain.clear();
	map<int, map<string, float>> word_to_class;
	const int kHistSize = hists.begin()->second.cols;
	gain.resize(kHistSize);
	
	map<string, float> class_probability;
	const int kNumClasses = hists.size();  //sub_directories.size();
	
	// initialize words data structure
	cout << "initialize words data structure\n";
	for (int i = 0; i < kHistSize; ++i) {
		for (map<string, cv::Mat>::const_iterator class_it = hists.begin(); class_it != hists.end(); ++class_it) {  // int j = 0; j < kNumClasses; ++j) {
			word_to_class[i][class_it->first] = 0.0f;
		}
	}

	// go through each file and keep track of what words are present
	//  also, calculate global entropy
	cout << "calculate word counts in images and global entropy.\n";
	int file_count = 0;
	for (map<string, cv::Mat>::iterator it = hists.begin(); it != hists.end(); ++it) {
		const string kClass = it->first;
		class_probability[kClass] = it->second.rows;  // # images in class
		file_count += it->second.rows;
		cout << "class : " << kClass << endl;
			
		for (int i = 0; i < it->second.rows; ++i) {
			for (int word_idx = 0; word_idx < kHistSize; ++word_idx) {
				if (0.0 < it->second.row(i).at<float>(word_idx))  // if words are present
					word_to_class[word_idx][kClass]++;
			}
		}
	}

	// normalize class file counts
	cout << "normalize class file counts.\n";
	cout << "  file_count = " << file_count << endl;
	for (map<string, float>::iterator it = class_probability.begin(); it != class_probability.end(); ++it) {
		cout << it->first << " = " << it->second << " / " << file_count << endl;
			it->second /= file_count;
	}

	// calculate the entropy of the entire training set
	const float kOriginalEntropy = entropy(class_probability);
	cout << "entropy of whole set is : " << kOriginalEntropy << endl;

	// calculate the gain of each word
	cout << "calculate gain for each word";
	for (map<int, map<string, float>>::iterator word_it = word_to_class.begin(); word_it != word_to_class.end(); ++word_it) {
		cout << " " << word_it->first << " ";
		const int kCurrentWord = word_it->first;
		float class_count = 0.0f;
		for (map<string, float>::iterator class_it = word_it->second.begin(); class_it != word_it->second.end(); ++class_it) {
			class_count += class_it->second;
		}

		// normalize class occurence counter for images that contain current word
		// This gives us the posterior probability for each class, given the current word is detected.
		if (0.0 < class_count) {
			for (map<string, float>::iterator class_it = word_it->second.begin(); class_it != word_it->second.end(); ++class_it) {
				class_it->second /= class_count;
			}

			// calculate entropy of data set where word is present
			const float kWordEntropy = entropy(word_it->second);

			// calculate reduction in entropy when word is present and store as information gain
			gain.at(kCurrentWord) = kOriginalEntropy - kWordEntropy;
		}
	}	
	cout << endl;
}

void calculateGainForAll(path dir) {
	// get list of all class subfolders
	std::vector<path> sub_directories;
	listSubDirectories(dir, sub_directories);
	cout << sub_directories.size() << " sub directories.\n";

	// go through each sub-folder, reading in histograms for all images present
	map<string, cv::Mat> surf_hists;
	map<string, cv::Mat> lab_hists;
	for (int dir_idx = 0; dir_idx < sub_directories.size(); ++dir_idx) {
		const path kCurrDir = sub_directories.at(dir_idx);
		cout << "reading histograms from " << kCurrDir << endl;
		cout << "  basename " << basename(kCurrDir) << endl;

		// get histograms from file
		//  SURF
		surf_hists[basename(kCurrDir)] = readMatFromFile(kCurrDir / "surf_hists.yml", kSurfHist);

		//  Lab
		lab_hists[basename(kCurrDir)] = readMatFromFile(kCurrDir / "lab_hists.yml", kLabHist);
	}


	// calculate information gain
	// SURF
	cout << "calc SURF information gain.\n";
	vector<float> surf_gain;
	calcHistGainSubdirectories(surf_hists, surf_gain);

	// Lab
	cout << "calc lab histogram information gain.\n";
	vector<float> lab_gain;
	calcHistGainSubdirectories(lab_hists, lab_gain);
	
	// write gain values to file
	// SURF
	cout << "writing SURF gain values to " << dir / "surf_gain.yml" << endl;
	FileStorage fs;
	fs.open((dir / "surf_gain.yml").string(), FileStorage::WRITE);
	fs << "surf_gain" << surf_gain;
	fs.release();

	// color
	cout << "writing Lab gain values to " << dir / "lab_gain.yml" << endl;
	fs.open((dir / "lab_gain.yml").string(), FileStorage::WRITE);
	fs << "lab_gain" << lab_gain;
	fs.release();
}

void searchGain(path search_dir, path query_img, std::vector<std::string> &results) {
	bool use_surf = false;
	
	// extract gain in image
	const float kSurfGain = getSurfGain(search_dir, query_img);
	const float kLabGain = getLabGain(search_dir, query_img);

	// decide whether to use SURF or color search
	cout << "surf gain: " << kSurfGain << endl;
	cout << "lab gain: " << kLabGain << endl;
	if (kSurfGain > kLabGain)
		use_surf = true;
	cout << "use_surf: " << use_surf << endl;
	// perform search
	if (use_surf)
		searchSURFHists(search_dir, query_img, results);
	else
		searchLab(search_dir, query_img, results);
	cout << "results.size() : " << results.size() << endl;
}

float getTotalGain(Mat hist, vector<float>& gain_values) {
	float total_gain = 0.0f;
	for (unsigned int i = 0; i < hist.cols; ++i) {
		total_gain += (hist.at<float>(i) > 0.005f) ? gain_values.at(i) : 0.0f;
	}
	return total_gain;
}

float getSurfGain(path search_dir, path img_path) {
	// read in gain values
	vector<float> gain_values;
	FileStorage fs;
	fs.open((search_dir / "surf_gain.yml").string(), FileStorage::READ);
	fs["surf_gain"] >> gain_values;

	// extract surf vocabulary histogram
	cout << "extracting surf histogram.\n";
	cv::Mat hist = extractVocabHistogram(img_path, search_dir / "surf_data.yml");

	// return information gain present in image histogram
	return getTotalGain(hist, gain_values);
}

float getHSVGain(path search_dir, path img_path) {
	// read in gain values
	vector<float> gain_values;
	FileStorage fs;
	fs.open((search_dir / "color_gain.yml").string(), FileStorage::READ);
	fs["color_gain"] >> gain_values;

	cv::Mat img = imread(img_path.string());

	cout << "extracting color histogram from image.\n";
	
	// extract color histogram from query image
	cvtColor(img, img, CV_BGR2HSV);
	cv::Mat hist = extractHSVHistogram(img);

	// return information gain present in image histogram
	return getTotalGain(hist, gain_values);
}

float getLabGain(path search_dir, path img_path) {
	// read in gain values
	vector<float> gain_values;
	FileStorage fs;
	fs.open((search_dir / "lab_gain.yml").string(), FileStorage::READ);
	fs["lab_gain"] >> gain_values;

	const Mat kVocab = readMatFromFile(search_dir / "lab_data.yml", kLabVocab);
	cv::Mat img = imread(img_path.string(), CV_LOAD_IMAGE_COLOR);

	cout << "extracting lab histogram from image.\n";
	
	// extract lab histogram from query image
	cvtColor(img, img, CV_BGR2Lab);
	cv::Mat hist = extractLabHistogram(img, kVocab);

	// return information gain present in image histogram
	return getTotalGain(hist, gain_values);
}
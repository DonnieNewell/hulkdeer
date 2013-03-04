#ifndef OPENCVTEST_CBIR_H_
#define OPENCVTEST_CBIR_H_

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2\flann\flann.hpp>
#include <opencv2\ml\ml.hpp>
#include <string>
#include <vector>

using namespace boost::filesystem;

const std::string kVocab("vocabulary");
const std::string kSearchIndex("search_index");
const std::string kSearchInvert("search_inverted");
const std::string kIndex("index");
const std::string kHist("histograms");
const std::string kSurfHist("surf_histograms");
const std::string kCountLines("count_lines");
const std::string kBuildLineClassifier("build_line_classifier");
const std::string kColorHistograms("color_hist");
const std::string kSearchColor("search_color");
const std::string kSearchSURF("search_surf");
const std::string kSearchDecide("search_decide");
const std::string kTestClassifier("test_classifier");

float calcHarmonicMean(cv::Mat &data);
cv::Mat extractTrainingVocabulary(path training_dir);
cv::Mat extractVocabHistograms(path training_dir, path vocab_file);
cv::Mat extractColorHistograms(path training_dir);
cv::Mat extractHSVHistogram(cv::Mat img);
float extractLineDescriptor(const cv::Mat kImg, cv::Mat &desc);
cv::flann::Index generateSearchIndex(cv::Mat vocab_hist);
cv::Mat readMatFromFile(const path kFilepath, const std::string kKey);
void writeMatToFile(const path kFilepath, const cv::Mat kData, const std::string kKey);
void writeClassifierToFile(const path kFilepath, const CvSVM kSVM, const std::string kKey);
CvSVM readClassifierFromFile(const path kFilepath, const std::string kKey);
void listDir(path dir, std::vector<path>& vec);
void listImgs(path dir, std::vector<path>& vec);
std::string getClass(std::string filename);
void searchIndex(path index_dir, path query_img);
void searchInvert(path index_dir, path query_img);
void displayResults(std::string query_filename, std::vector<std::string> &filenames);
std::vector<std::vector<int>> createInvertedFileList(cv::Mat histograms);
void countLines(path img_dir, std::map<std::string, cv::Mat> &line_descriptors, std::vector<std::string> &class_names);
float calcStructure(const cv::Mat kImg);
float calcEntropy(cv::Mat img);
float calcLineEntropy(const std::list<int> &kLineIndices, const std::vector<cv::Vec4i> &kLines);
void getEntropyMap(const cv::Mat &kImg, std::vector<cv::Vec4i> &lines, cv::Mat &map);
float entropy(const std::vector<float> &kHist);
float entropy(const cv::Mat kHist);
float computeShannonEntropy(const cv::Mat kHist);
float getHistogramBinValue(cv::Mat hist, int binNum);
float getFrequencyOfBin(cv::Mat channel);
void getLongerLines(const std::vector<cv::Vec4i> &kLines, const float kMinLength,
	std::vector<cv::Vec4i> &long_lines);
float getDistance(const cv::Vec4i kLine);
cv::Vec2i getMidPoint(const cv::Vec4i kLine);
void getCoterminations(const std::vector<cv::Vec4i> &kLines, const float kSimilarityAngle,
	const float kDistanceThreshold, std::vector<cv::Vec4i> &coterm_lines);
void getLJunctions(const std::vector<cv::Vec4i> &kLines, const float kDeltaLAngle,
	const float kDistanceThreshold, std::vector<cv::Vec4i> &l_junct_lines);
float getAngleBetweenLines(const cv::Vec4i kLine1, const cv::Vec4i kLine2);
float getDiffY(const cv::Vec4i kLine1, const cv::Vec4i kLine2);
float getDiffX(const cv::Vec4i kLine1, const cv::Vec4i kLine2);
void getParallelGroups(const std::vector<cv::Vec4i> &kParallelLines, const float kSimilarityAngle,
	const float kLengthRatio, const float kDistanceThreshold, const float kOverlapThreshold,
	std::vector<cv::Vec4i> &parallel_groups);
void getParallelLines(const std::vector<cv::Vec4i> &kLines, const float kSimilarityAngle,
	std::vector<cv::Vec4i> &parallel_lines);
float getDistanceRatio(const cv::Vec4i kLine1, const cv::Vec4i kLine2);
float getOverlapRatio(const cv::Vec4i kLine1, const cv::Vec4i kLine2);
bool isLeftOfLine(const cv::Vec4i kLine, cv::Vec2i kPoint);
void getUJunctions(const std::vector<cv::Vec4i> &kLines, const float kDeltaLAngle,
		const float kDistanceThreshold, std::vector<cv::Vec4i> &u_junct_lines);
inline void swap(int &val1, int &val2) { val1 ^= val2; val2 ^= val1; val1 ^= val2; }
void buildClassifiers(path p, std::map<std::string, cv::SVM> &classifiers);
void removeNoisyLines(const cv::Mat &kImg, std::vector<cv::Vec4i> &lines);
bool intersection(cv::Vec4i line_1, cv::Vec4i line_2, cv::Point2f &r);
void searchColor(path index_dir, path query_img, std::vector<std::string> &results);
void searchSURFHists(path index_dir, path query_img, std::vector<std::string> &results);
void searchDecideSURFColor(path index_dir, path query_img, const float kThreshold, std::vector<std::string> &results);
#endif
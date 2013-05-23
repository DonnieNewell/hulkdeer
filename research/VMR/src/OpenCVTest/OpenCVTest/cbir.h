#ifndef OPENCVTEST_CBIR_H_
#define OPENCVTEST_CBIR_H_

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2\flann\flann.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <string>
#include <vector>
#include <map>

using namespace boost::filesystem;

const std::string kVocab("vocabulary");
const std::string kLabVocab("lab_vocab");
const std::string kSearchIndex("search_index");
const std::string kSearchInvert("search_inverted");
const std::string kIndex("index");
const std::string kHist("histograms");
const std::string kSurfHist("surf_histograms");
const std::string kLabHist("lab_histograms");
const std::string kGaborResponse("gabor_response");
const std::string kCountLines("count_lines");
const std::string kBuildLineClassifier("build_line_classifier");
const std::string kColorHistograms("color_hist");
const std::string kSearchColor("search_color");
const std::string kSearchLab("search_lab");
const std::string kSearchSURF("search_surf");
const std::string kSearchGabor("search_gabor");
const std::string kSearchGain("search_gain");
const std::string kSearchDecide("search_decide");
const std::string kCalcGain("calc_gain");
const std::string kTestClassifier("test_classifier");
const std::string kTestSearch("test_search");
const std::string kCalcClassPrecision("calc_class_precision");
const std::string kCalcClassPrecisionGain("calc_class_precision_gain");
const std::string kRPrecisionCSV("r_precision_csv");


float calcHarmonicMean(cv::Mat &data);
cv::Mat extractTrainingVocabulary(path training_dir);
cv::Mat extractLabVocabulary(path training_dir);
void extractVocabHistograms(path img_dir, path vocab_file, std::vector<cv::Mat>& histograms);
cv::Mat extractVocabHistogram(path img_path, path vocab_file);
void extractColorHistograms(path training_dir, std::vector<cv::Mat>& histograms);
cv::Vec3f extractCommonLabColor(cv::Mat img);
cv::Mat extractCommonLabColors(cv::Mat img);
cv::Mat extractHSVHistogram(cv::Mat img);
cv::Mat extractLabHistogram(cv::Mat img, cv::Mat vocab);
void extractLabHistograms(path img_dir, path vocab_file, std::vector<cv::Mat>& histograms);
float extractLineDescriptor(const cv::Mat kImg, cv::Mat &desc);
cv::flann::Index generateSearchIndex(cv::Mat vocab_hist);
cv::Mat readMatFromFile(const path kFilepath, const std::string kKey);
void writeMatToFile(const path kFilepath, const cv::Mat kData, const std::string kKey);
void writeClassifierToFile(const path kFilepath, const CvSVM kSVM, const std::string kKey);
CvSVM readClassifierFromFile(const path kFilepath, const std::string kKey);
void listDir(path dir, std::vector<path>& vec);
void listSubDirectories(path dir, std::vector<path>& sub_directories);
void listImgs(path dir, std::vector<path>& vec);
void listSubDirImgs(path dir, std::vector<path>& vec);
std::string getClass(std::string filename);
std::string getClass(const unsigned int kIndex);
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
float entropy(const std::map<std::string, float> &kHist);
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
void searchLab(path index_dir, path query_img, const int kK, std::vector<std::string> &results);
void searchLab(cv::Mat vocab, cv::Mat hists, std::vector<path> image_names, path query_img, const int kK, std::vector<std::string> &results);
void searchSURFHists(path index_dir, path query_img, const int kNN, std::vector<std::string> &results);
void searchSURFHists(cv::Mat vocab, cv::Mat hists, std::vector<path> image_names, std::vector<path> query_imgs, const int kNN, std::vector<std::vector<cv::DMatch>> &results);
void searchSURFHists(cv::Mat query_hists, cv::Mat search_hists, std::vector<path> image_names, const int kNN, std::vector<std::vector<cv::DMatch>> &results);
void searchDecideSURFColor(path index_dir, path query_img, const float kThreshold, std::vector<std::string> &results);
void searchGain(path search_dir, path query_img, const int kK, std::vector<std::string> &results);
void calcHistGain(std::vector<path>& filenames, cv::Mat& hists, std::vector<float>& gain);
template<typename T, size_t N>
T * my_end(T (&ra)[N]) {
    return ra + N;
}
std::string getClassFolder(const std::string kFilePath);
void calculateGainForAll(path dir);
void calcPrecisionVector(const std::vector<path> &kImageNames, const int kQueryStartIdx, const int kNumQueries, const std::vector<std::vector<cv::DMatch>>& kResults, std::vector<float>& precision);
void calcPrecisionVector(const std::string kQueryName, const std::vector<std::string>& kResults, std::vector<float>& precision);
void calcPrecisionVector(const std::vector<path> &kDatabaseImages, const std::vector<path> &kQueryImages, const std::vector<std::vector<cv::DMatch>>& kResults, std::vector<float>& precision);
void calcRecallVector(const std::vector<path> kImageNames, const int kQueryStartIdx, const int kNumQueries, const std::vector<std::vector<cv::DMatch>>& kResults, std::vector<float>& recall);
void calcRecallVector(const path kDir, const std::string kQueryName, const std::vector<std::string>& kResults, std::vector<float>& recall);
float getTotalGain(cv::Mat hist, std::vector<float>& gain_values);
float getSurfGain(path search_dir, path img_path);
float getHSVGain(path search_dir, path img_path);
float getLabGain(path search_dir, path img_path);
void calcHistGainSubdirectories(/*vector<path>& sub_directories, */std::map<std::string, cv::Mat>& hists, std::vector<float>& gain);
void writePrecisionRecallCSV(const std::vector<float>& kPrecision, const std::vector<float>& kRecall, const std::string kFilename);
void testSearch(const path kDir, const std::string kSearchMode, std::vector<float>& precision, std::vector<float>& recall);
void loadHists(const path kDir, const std::string kFileName, const std::string kDataKey, cv::Mat& hists);
void loadHists(const path kDir, const std::vector<std::string>& kSearchModes, std::vector<cv::Mat>& hists);
void calcPrecisionAllClasses(const path kQueryDir, const path kDir, const std::string kSearchMode);
void calcPrecisionAllClassesGain(const path kQueryDir, const path kDir, const std::vector<std::string> kSearchModes);
void getDataFilenameAndKey(const std::string kSearchMode, std::string &data_file, std::string &data_key);
void searchGenericHists(cv::Mat query_hists, cv::Mat database, const int kNN, std::vector<std::vector<cv::DMatch>> &results);
void collectRPrecisionData(const path kDir, const std::string kSearchMode, std::vector<float> &r_precision);
void writeToCSV(const std::vector<float> &kData, std::string filename);
void getGainFilenameAndKey(const std::string kSearchMode, std::string &data_file, std::string &data_key);
#endif
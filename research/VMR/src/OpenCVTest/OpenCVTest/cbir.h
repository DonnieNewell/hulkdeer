#ifndef OPENCVTEST_CBIR_H_
#define OPENCVTEST_CBIR_H_

/** 
* @file cbir.h 
* @brief this header file will contain all required 
* definitions and basic utilities functions associated
* with the VMR project.
*
* @author Donald Newell
*
* @date 5/24/2013
*/

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2\flann\flann.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <string>
#include <vector>
#include <map>

using namespace boost::filesystem;

/** generates vocabulary for bag-of-words */
const std::string kVocab("vocabulary");

/** generates vocabulary for bag-of-colors */
const std::string kLabVocab("lab_vocab");

/** searches database using pre-computed index */
const std::string kSearchIndex("search_index");

/** searches inverted file list */
const std::string kSearchInvert("search_inverted");

/** generates search index */
const std::string kIndex("index");

/** flag used as key to read some histograms from yml files */
const std::string kHist("histograms");

/** key for reading SURF histograms from yml files */
const std::string kSurfHist("surf_histograms");

/** key for reading Lab histograms from yml files */
const std::string kLabHist("lab_histograms");

/** key for reading Gabor responses from yml files */
const std::string kGaborResponse("gabor_response");

/** command-line flag for generating line structure descriptors */
const std::string kCountLines("count_lines");

/** command-line flag for creating SVM classifiers for image classes
based on previously extracted line structure descriptors. */
const std::string kBuildLineClassifier("build_line_classifier");

/** command-line flag for generating HSV histogram data */
const std::string kColorHistograms("color_hist");

/** command-line flag for searching images by HSV */
const std::string kSearchColor("search_color");

/** command-line flag for searching images using CIE-Lab color space */
const std::string kSearchLab("search_lab");

/** command-line flag for searching images using SURF descriptors in Opponent Color Space */
const std::string kSearchSURF("search_surf");

/** command-line flag for searching images using Gabor descriptors */
const std::string kSearchGabor("search_gabor");

/** command-line flag for searching the image database using information
gain self-nomination */
const std::string kSearchGain("search_gain");

/** command-line flag for searching image database using line structure as 
threshold for deciding between HSV and SURF descriptors. */
const std::string kSearchDecide("search_decide");

/** command-line flag for calculating the gain for an image database. */
const std::string kCalcGain("calc_gain");

/** command-line flag for measuring the performance of the SVM classifier
that uses line structure descriptors */
const std::string kTestClassifier("test_classifier");

/** command-line flag for calculating Precision and recall of the specified
search method */
const std::string kTestSearch("test_search");

/** command-line flag for calculating the R-Precision for each class of image,
 using the specified search algorithm */
const std::string kCalcClassPrecision("calc_class_precision");

/** command-line flag for calculating the R-Precision for each class of image,
 using the method with the highest information gain */
const std::string kCalcClassPrecisionGain("calc_class_precision_gain");

/** command-line flag for compiling the pre-existing R-Precision data files
into a csv file for creating graphs. */
const std::string kRPrecisionCSV("r_precision_csv");

/** @brief Calculates the harmonic mean of the matrix 
@param data matrix for which the harmonic mean is calculated
*/
float calcHarmonicMean(cv::Mat &data);

/** extracts the SURF features from the images in the training_dir and generates the vocabulary */
cv::Mat extractTrainingVocabulary(path training_dir);

/** extracts the Lab features from the images in the training_dir and generates the color vocabulary */
cv::Mat extractLabVocabulary(path training_dir);

/** uses the specified vocab file to extract the histograms from all of the images in the img_dir */
void extractVocabHistograms(path img_dir, path vocab_file, std::vector<cv::Mat>& histograms);

/** uses the specified vocab file to extract the SURF Bag-of-Words histogram from the specified img_path */
cv::Mat extractVocabHistogram(path img_path, path vocab_file);

/** extracts the HSV histograms from the images in the specified directory. */
void extractColorHistograms(path training_dir, std::vector<cv::Mat>& histograms);

/** returns the most common Lab color in the img */
cv::Vec3f extractCommonLabColor(cv::Mat img);

/** returns the most common color for each of 256 regions in the image. */
cv::Mat extractCommonLabColors(cv::Mat img);

/** returns the HSV histogram for the image */
cv::Mat extractHSVHistogram(cv::Mat img);

/** returns the CIE-Lab histogram for the specified image using the specified color vocabulary. */
cv::Mat extractLabHistogram(cv::Mat img, cv::Mat vocab);

/** extracts the CIE-Lab histograms for all of the images in the specified directory. */
void extractLabHistograms(path img_dir, path vocab_file, std::vector<cv::Mat>& histograms);

/** extracts the line structure descriptor from the image */
float extractLineDescriptor(const cv::Mat kImg, cv::Mat &desc);

/** creates a search index using the specified histograms */
cv::flann::Index generateSearchIndex(cv::Mat vocab_hist);

/** opens the yml file specified and returns the matrix associated
with the specified key. */
cv::Mat readMatFromFile(const path kFilepath, const std::string kKey);

/** writes the specified data matrix to the filename specified and
associates the matrix with the specified key. */
void writeMatToFile(const path kFilepath, const cv::Mat kData, const std::string kKey);

/** writes the classifier parameters to file associated with the specified key. */
void writeClassifierToFile(const path kFilepath, const CvSVM kSVM, const std::string kKey);

/** reads the svm parameters from the specified file using the associated key. */
CvSVM readClassifierFromFile(const path kFilepath, const std::string kKey);

/** provides a list of all of the files in the specified directory. */
void listDir(path dir, std::vector<path>& vec);

/** lists all of the subdirectories in the specified directory */
void listSubDirectories(path dir, std::vector<path>& sub_directories);

/** lists all of the images in the specified directory */
void listImgs(path dir, std::vector<path>& vec);

/** lists all of the images present in all of the subdirectories of
the specified directory */
void listSubDirImgs(path dir, std::vector<path>& vec);

/** returns the image class based off of the old VMR image dataset */
std::string getClass(std::string filename);

/** returns the image class based off of the class index using the old VMR image classes */
std::string getClass(const unsigned int kIndex);

/** searches the specified index using SURF Bag-of-words histogram from the specified image */
void searchIndex(path index_dir, path query_img);

/** searches the specified inverted file list using SURF Bag-of-words histogram from the specified image */
void searchInvert(path index_dir, path query_img);

/** displays the query image and then displays the result images one-by-one. */
void displayResults(std::string query_filename, std::vector<std::string> &filenames);

/** creates an inverted file list using the SURF histograms for each image in the database. */
std::vector<std::vector<int>> createInvertedFileList(cv::Mat histograms);

/** returns line structure descriptors and class names for the images in the specified directory */
void countLines(path img_dir, std::map<std::string, cv::Mat> &line_descriptors, std::vector<std::string> &class_names);

/** returns the structure value for the specified image */
float calcStructure(const cv::Mat kImg);

/** calculates the entropy of the pixel values in the image. */
float calcEntropy(cv::Mat img);

/** calculates the entropy of the line angles. */
float calcLineEntropy(const std::list<int> &kLineIndices, const std::vector<cv::Vec4i> &kLines);

/** returns the line-angle entropy map of the image using the specified lines. */
void getEntropyMap(const cv::Mat &kImg, std::vector<cv::Vec4i> &lines, cv::Mat &map);

/** calculates the entropy of the input histogram */
float entropy(const std::vector<float> &kHist);

/** calculates the entropy of the input histogram */
float entropy(const cv::Mat kHist);

/** calculates the entropy of the input histogram */
float entropy(const std::map<std::string, float> &kHist);

/** calculates the entropy of the input histogram */
float computeShannonEntropy(const cv::Mat kHist);

/** returns the floating point value of the specified element in the matrix */
float getHistogramBinValue(cv::Mat hist, int binNum);

/** returns the sum of all of the elements in the histogram. */
float getFrequencyOfBin(cv::Mat channel);

/** returns all lines longer than the specified threshold. */
void getLongerLines(const std::vector<cv::Vec4i> &kLines, const float kMinLength,
	std::vector<cv::Vec4i> &long_lines);

/** returns the Euclidean distance of the line. */
float getDistance(const cv::Vec4i kLine);

/** returns the mid point along the specified line. */
cv::Vec2i getMidPoint(const cv::Vec4i kLine);

/** returns all lines that have a relative angle greater than the similarity angle, 
  and have endpoints within the specified distance threshold of each other. */
void getCoterminations(const std::vector<cv::Vec4i> &kLines, const float kSimilarityAngle,
	const float kDistanceThreshold, std::vector<cv::Vec4i> &coterm_lines);

/** returns all lines that are in L-junctions as specified by the angle and distance threshold. */
void getLJunctions(const std::vector<cv::Vec4i> &kLines, const float kDeltaLAngle,
	const float kDistanceThreshold, std::vector<cv::Vec4i> &l_junct_lines);

float getAngleBetweenLines(const cv::Vec4i kLine1, const cv::Vec4i kLine2);

/** returns the smallest difference in the y-direction between 2 endpoints out of the 4. */
float getDiffY(const cv::Vec4i kLine1, const cv::Vec4i kLine2);

/** returns the smallest difference in the x-direction between 2 endpoints out of the 4. */
float getDiffX(const cv::Vec4i kLine1, const cv::Vec4i kLine2);

/** returns all lines that are in parallel groups as specified by relative distance and
length of line projected onto each other. */
void getParallelGroups(const std::vector<cv::Vec4i> &kParallelLines, const float kSimilarityAngle,
	const float kLengthRatio, const float kDistanceThreshold, const float kOverlapThreshold,
	std::vector<cv::Vec4i> &parallel_groups);

/** returns all lines that are parallel to other lines */
void getParallelLines(const std::vector<cv::Vec4i> &kLines, const float kSimilarityAngle,
	std::vector<cv::Vec4i> &parallel_lines);

/** returns a ratio that represents the distance between lines relative to line length */
float getDistanceRatio(const cv::Vec4i kLine1, const cv::Vec4i kLine2);

/** returns the ratio of the line projection onto another line relative to it's length */
float getOverlapRatio(const cv::Vec4i kLine1, const cv::Vec4i kLine2);

/** returns true if the point is "left" of the line. */
bool isLeftOfLine(const cv::Vec4i kLine, cv::Vec2i kPoint);

/** returns all lines that are a part of U-junctions, as specified by the input parameters. */
void getUJunctions(const std::vector<cv::Vec4i> &kLines, const float kDeltaLAngle,
		const float kDistanceThreshold, std::vector<cv::Vec4i> &u_junct_lines);

/** efficient integer swap function using XOR */
inline void swap(int &val1, int &val2) { val1 ^= val2; val2 ^= val1; val1 ^= val2; }

/** creates SVM classifiers for each class using line structure descriptors */
void buildClassifiers(path p, std::map<std::string, cv::SVM> &classifiers);

/** looks at line angle entropy in different regions of the image and removes 
all lines that fall in regions that have a line angle entropy above a threshold. */
void removeNoisyLines(const cv::Mat &kImg, std::vector<cv::Vec4i> &lines);

/** find the intersection point of the lines. return false if no intersection. */
bool intersection(cv::Vec4i line_1, cv::Vec4i line_2, cv::Point2f &r);

/** search using HSV */
void searchColor(path index_dir, path query_img, std::vector<std::string> &results);

/** search the specified directory  using the query image and return the k closest matches. */
void searchLab(path index_dir, path query_img, const int kK, std::vector<std::string> &results);

/** searches through the pre-extracted histograms in the CIE-Lab color space, using bag-of-colors */
void searchLab(cv::Mat vocab, cv::Mat hists, std::vector<path> image_names, path query_img, const int kK, std::vector<std::string> &results);

/** loads the surf histograms and vocabulary from the directory, extracts the surf histogram from the query image, and stores the K nearest neighbors in the results vector. */
void searchSURFHists(path index_dir, path query_img, const int kNN, std::vector<std::string> &results);

/** uses the supplied vocabulary to extract the surf histograms from the specified images, and then search through the histograms for the closest matches. */
void searchSURFHists(cv::Mat vocab, cv::Mat hists, std::vector<path> image_names, std::vector<path> query_imgs, const int kNN, std::vector<std::vector<cv::DMatch>> &results);

/** uses all of the pre-extracted histograms to find the K closest images for each search histogram and stores it in the results vector. */
void searchSURFHists(cv::Mat query_hists, cv::Mat search_hists, std::vector<path> image_names, const int kNN, std::vector<std::vector<cv::DMatch>> &results);

/** Uses the structure metric to decide whether to search with SURF or Lab color histograms, using the specified threshold. */
void searchDecideSURFColor(path index_dir, path query_img, const float kThreshold, std::vector<std::string> &results);

/** loads the histograms, vocabularies, and gain values from the search directory.
Extracts the histograms from the query image, and then uses the sum of the gain values for the 
detected features/colors to determine which descriptor technique provides the most information. */
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
void collectRPrecisionData(const path kQueryDir, const path kDir, const std::string kSearchMode, std::vector<float> &r_precision);
void writeToCSV(const std::vector<float> &kData, std::string filename);
void getGainFilenameAndKey(const std::string kSearchMode, std::string &data_file, std::string &data_key);
#endif
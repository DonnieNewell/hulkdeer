// This is the main DLL file.

#include "Gist.h"

using namespace Gist;

/*! Graylevel GIST for various scales. 
* Descriptor size is  kNBlocks*kNBlocks*sum(kNOrientations[i],i=0..kNScale-1)
* wrapper on leargist implementation.	 
*/
float* LearGist::GrayscaleGistScaletab(image_t* src, const int kNBlocks, const int kNScale, const int* kNOrientations) {
	float* result = 0;
	result = bw_gist_scaletab(src, kNBlocks, kNScale, kNOrientations);
	return result;
}

/*! Graylevel GIST. 
* Descriptor size is kNBlocks*kNBlocks*(kA+kB+kC)
* wrapper on leargist implementation.	 
*/
float* LearGist::GrayscaleGist(image_t* src, const int kNBlocks, const int kA, const int kB, const int kC) {
	float* result = 0;
	result = bw_gist(src, kNBlocks, kA, kB, kC);
	return result;
}

/*! Color GIST for various scales. 
* Descriptor size is kNBlocks*kNBlocks*(kA+kB+kC)
* wrapper on leargist implementation.	 
*/
float* LearGist::ColorGist(color_image_t* src, const int kNBlocks, const int kA, const int kB, const int kC) {
	float* result = 0;
	result = color_gist(src, kNBlocks, kA, kB, kC);
	return result;
}

/*! Color GIST. 
* Descriptor size is  kNBlocks*kNBlocks*sum(kNOrientations[i],i=0..kNScale-1)
* wrapper on leargist implementation.	 
*/
float* LearGist::ColorGistScaletab(color_image_t* src, const int kNBlocks, const int kNScale, const int* kNOrientations) {
	float* result = 0;
	result = color_gist_scaletab(src, kNBlocks, kNScale, kNOrientations);
	return result;
}
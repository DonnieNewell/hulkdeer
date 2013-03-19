// Gist.h
// Donnie Newell (den4gr)

#pragma once

#include "lear_gist.h"

namespace Gist {

	class LearGist
	{
		/*! Wrapper for leargist C library. http://lear.inrialpes.fr/software */
	public:
		/*! Graylevel GIST for various scales. 
		 * Descriptor size is  kNBlocks*kNBlocks*sum(kNOrientations[i],i=0..kNScale-1)
		 * wrapper on leargist implementation.	 
		 */
		float* GrayscaleGistScaletab(image_t* src, const int kNBlocks, const int kNScale, const int* kNOrientations);
		
		/*! Graylevel GIST. 
		 * Descriptor size is kNBlocks*kNBlocks*(kA+kB+kC)
		 * wrapper on leargist implementation.	 
		 */
		float* GrayscaleGist(image_t* src, const int kNBlocks, const int kA, const int kB, const int kC);
		
		/*! Color GIST for various scales. 
		 * Descriptor size is kNBlocks*kNBlocks*(kA+kB+kC)
		 * wrapper on leargist implementation.	 
		 */
		float* ColorGist(color_image_t* src, const int kNBlocks, const int kA, const int kB, const int kC);
		
		/*! Color GIST. 
		 * Descriptor size is  kNBlocks*kNBlocks*sum(kNOrientations[i],i=0..kNScale-1)
		 * wrapper on leargist implementation.	 
		 */
		float* ColorGistScaletab(color_image_t* src, const int kNBlocks, const int kNScale, const int* kNOrientations);
	};
}

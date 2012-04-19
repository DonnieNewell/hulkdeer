#include <stdio.h>
#include <jpeglib.h>
#include <stdlib.h>
#include <math.h>
#include <npp.h>

#define DEBUG 1

/* we will be using this uninitialized pointer later to store raw, uncompressd image */
//unsigned char *raw_image = NULL;

/* dimensions of the image we want to write */
int width = 1600;
int height = 1200;
int bytes_per_pixel = 3;   /* or 1 for GRAYSCALE images */
J_COLOR_SPACE color_space = JCS_RGB; /* or JCS_GRAYSCALE for grayscale images */

void rgb2gray(const unsigned char *src, double*dst, size_t width, size_t height){
	for(int i = 0; i<height; i++){
		for(int j = 0; j < width; j++){
			int index = (i*width + j);
				//grayscale
				dst[index] = .30*src[index*bytes_per_pixel] + .59*src[index*bytes_per_pixel+1] + .11*src[index*bytes_per_pixel+2]; 
	}
	}	
}

void convert_to_integral(double *src, size_t width, size_t height){
	for(int i=0; i<height; i++){
		for(int j =0; j < width; j++){
			int index = i*width + j;
			double south = 0.0, west = 0.0;
			if(j > 0) west = src[index-1];
			if(i > 0) south = src[index-width];
			src[index]+= west+south;
		}
	}
}

/**
 * read_jpeg_file Reads from a jpeg file on disk specified by filename and saves into the 
 * raw_image buffer in an uncompressed format.
 * 
 * \returns positive integer if successful, -1 otherwise
 * \param *filename char string specifying the file name to read from
 *
 */

int read_jpeg_file(unsigned char **raw_image, char *filename, int &width, int &height )
{
	/* these are standard libjpeg structures for reading(decompression) */
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	/* libjpeg data structure for storing one row, that is, scanline of an image */
	JSAMPROW row_pointer[1];
	
	FILE *infile = fopen( filename, "rb" );
	unsigned long location = 0;
	int i = 0;
	
	if ( !infile )
	{
		printf("Error opening jpeg file %s\n!", filename );
		return -1;
	}
	/* here we set up the standard libjpeg error handler */
	cinfo.err = jpeg_std_error( &jerr );
	/* setup decompression process and source, then read JPEG header */
	jpeg_create_decompress( &cinfo );
	/* this makes the library read from infile */
	jpeg_stdio_src( &cinfo, infile );
	/* reading the image header which contains image information */
	jpeg_read_header( &cinfo, TRUE );
	/* Uncomment the following to output image information, if needed. */
	width = cinfo.image_width;
	height = cinfo.image_height;
	bytes_per_pixel = cinfo.num_components;
	
	if(DEBUG){
		printf( "JPEG File Information: \n" );
		printf( "Image width and height: %d pixels and %d pixels.\n", cinfo.image_width, cinfo.image_height );
		printf( "Image outputwidth and outputheight: %d pixels and %d pixels.\n", cinfo.output_width, cinfo.output_height );
		printf( "Color components per pixel: %d.\n", cinfo.num_components );
		printf( "Color space: %d.\n", cinfo.jpeg_color_space );
	}
	/* Start decompression jpeg here */
	jpeg_start_decompress( &cinfo );

	
	/* allocate memory to hold the uncompressed image */
	*raw_image = (unsigned char*)malloc( cinfo.image_width*cinfo.image_height*cinfo.num_components );
	/* now actually read the jpeg into the raw buffer */
	row_pointer[0] = (unsigned char *)malloc( cinfo.image_width*cinfo.num_components );
	/* read one scan line at a time */
	while( cinfo.output_scanline < cinfo.image_height )
	{
		jpeg_read_scanlines( &cinfo, row_pointer, 1 );
		for( i=0; i<cinfo.image_width*cinfo.num_components;i++) 
			(*raw_image)[location++] = row_pointer[0][i];
	}
	/* wrap up decompression, destroy objects, free pointers and close open files */
	jpeg_finish_decompress( &cinfo );
	jpeg_destroy_decompress( &cinfo );
	free( row_pointer[0] );
	fclose( infile );
	/* yup, we succeeded! */
	return 1;
}

/**
 * write_jpeg_file Writes the raw image data stored in the raw_image buffer
 * to a jpeg image with default compression and smoothing options in the file
 * specified by *filename.
 *
 * \returns positive integer if successful, -1 otherwise
 * \param *filename char string specifying the file name to save to
 *
 */
int write_jpeg_file(unsigned char *raw_image, char *filename, int width, int height )
{
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	
	/* this is a pointer to one row of image data */
	JSAMPROW row_pointer[1];
	FILE *outfile = fopen( filename, "wb" );
	
	if ( !outfile )
	{
		printf("Error opening output jpeg file %s\n!", filename );
		return -1;
	}
	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);

	/* Setting the parameters of the output file here */
	cinfo.image_width = width;	
	cinfo.image_height = height;
	cinfo.input_components = bytes_per_pixel;
	cinfo.in_color_space = color_space;
    /* default compression parameters, we shouldn't be worried about these */
	jpeg_set_defaults( &cinfo );
	/* Now do the compression .. */
	jpeg_start_compress( &cinfo, TRUE );
	/* like reading a file, this time write one row at a time */
	while( cinfo.next_scanline < cinfo.image_height )
	{
		row_pointer[0] = &raw_image[ cinfo.next_scanline * cinfo.image_width *  cinfo.input_components];
		jpeg_write_scanlines( &cinfo, row_pointer, 1 );
	}
	/* similar to read file, clean up after we're done compressing */
	jpeg_finish_compress( &cinfo );
	jpeg_destroy_compress( &cinfo );
	fclose( outfile );
	/* success code is 1! */
	return 1;
}

/**
* prints command-line usage for program
*
*/
void usage(){
  printf("usage: ./jpeg_sample <jpeg_image_filename>\n");
}

unsigned char clamp_32f_to_8u(float val){
	if(255.0f < val) 
		return 255;
	if(0.0f > val)
		return 0;
	return (unsigned char) floor(val+0.5f);
}

/** returns non-zero on failure
*/
int convert_8uC3_to_32fC4(const unsigned char* p_src, size_t n_src_width, size_t n_src_height, float *p_dst){
	const int src_channels_per_pixel = 3;
	const int dst_channels_per_pixel = 4;
	int row;
	for(row=0; row<n_src_height; row++){
		int col;		
		for(col=0; col<n_src_width; col++){
			const size_t pix = (row*n_src_width+col);
			const size_t src_index = pix*src_channels_per_pixel;
			const size_t dst_index = pix*dst_channels_per_pixel;
			p_dst[dst_index] = (float)p_src[src_index];
			p_dst[dst_index+1] = (float)(p_src[src_index+1]);
			p_dst[dst_index+2] = (float)(p_src[src_index+2]);
			p_dst[dst_index+3] = 1.0f;
		}
	}

	return 0;
}

/** returns non-zero on failure
*/
int convert_32fC4_to_8uC3(const float* p_src, size_t n_src_width, size_t n_src_height, unsigned char *p_dst){
	const int src_channels_per_pixel = 4;
	const int dst_channels_per_pixel = 3;
	int row;
	for(row=0; row<n_src_height; row++){
		int col;		
		for(col=0; col<n_src_width; col++){
			const size_t pix = (row*n_src_width+col);
			const size_t src_index = pix*src_channels_per_pixel;
			const size_t dst_index = pix*dst_channels_per_pixel;
			p_dst[dst_index] = clamp_32f_to_8u(p_src[src_index]);
			p_dst[dst_index+1] = clamp_32f_to_8u(p_src[src_index+1]);
			p_dst[dst_index+2] = clamp_32f_to_8u(p_src[src_index+2]);
		}
	}

	return 0;
}

__global__ void non_maximal_suppression(double* src, int* points, int width, int height){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int n = 3;
	i = i*n;
	j = j*n;
	int index = (i*width+j);
	if(i < height && j < width){
		int mi = i;
		int mj = j;
		/* find maximum in current grid */
		for(int i2 = i; i2 <= i+n; i2++){
			for(int j2 = j; j2 <= j+n; j2++){
				if(src[i2*width+j2] > src[mi*width+mj]){
					mi = i2;
					mj = j2;
				}//end if
			}//end for j2
		} //end for i2

		/* check neighborhood around maximum */
		for(int i2 = mi-n; i2 <= mi+n; i2++){
			for(int j2 = mj-n; j2 <= mj+n; j2++){
	}//end if
	
}//end nms()
__global__ void calc_det_hessian(double* dImg, double* dDetHess, int width, int height){
	//calc index
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int filter_width = 9;
	int half_filter_width = filter_width/2;
	
	//check if entire filter would fit at this location
	if(i < (height-half_filter_width) && 
		i >= half_filter_width && 
		j < (width-half_filter_width) && 
		j >= half_filter_width){
		
		int lobe_height = filter_width/3;
		int lobe_width = 5;


		//Dyy
		// get middle lobe value
		int index = (i+lobe_height/2)*width+j+lobe_width/2;
		double tmp = dImg[index]-dImg[index-lobe_width]-dImg[index - lobe_height*width]+dImg[index-lobe_width- lobe_height*width];
		double Dyy = -2*tmp;
		
		//get upper lobe
		index -= (lobe_height)*width;
		tmp = dImg[index]-dImg[index-lobe_width]-dImg[index - lobe_height*width]+dImg[index-lobe_width- lobe_height*width];
		Dyy += tmp;

		//get lower lobe
		index += (2*lobe_height)*width;
		tmp = dImg[index]-dImg[index-lobe_width]-dImg[index - lobe_height*width]+dImg[index-lobe_width- lobe_height*width];
		Dyy += tmp;

		
		//Dxx
		// get middle lobe value
		index = (i+lobe_width/2)*width+j+lobe_height/2;
		tmp = dImg[index]-dImg[index-lobe_height]-dImg[index - lobe_width*width]+dImg[index-lobe_height- lobe_width*width];
		double Dxx = -2*tmp;
		
		//get left lobe
		index -= (lobe_height);
		tmp = dImg[index]-dImg[index-lobe_height]-dImg[index - lobe_width*width]+dImg[index-lobe_height- lobe_width*width];
		Dxx += tmp;

		//get right lobe
		index += (2*lobe_height);
		tmp = dImg[index]-dImg[index-lobe_height]-dImg[index - lobe_width*width]+dImg[index-lobe_height- lobe_width*width];
		Dxx += tmp;
		

		//Dxy
		//get upper left
		index = (i-1)*width+j-1;
		tmp = dImg[index] - dImg[index-lobe_height] - dImg[index-lobe_height*width] + dImg[index-lobe_height-lobe_height*width];
		double Dxy = tmp;
		
		//get upper right
		index += 1+lobe_height;
		tmp = dImg[index] - dImg[index-lobe_height] - dImg[index-lobe_height*width] + dImg[index-lobe_height-lobe_height*width];
		Dxy -= tmp;

		//get lower right
		index += (1+lobe_height)*width;
		tmp = dImg[index] - dImg[index-lobe_height] - dImg[index-lobe_height*width] + dImg[index-lobe_height-lobe_height*width];
		Dxy += tmp;
		
		//get lower left
		index -= 1+lobe_height;
		tmp = dImg[index] - dImg[index-lobe_height] - dImg[index-lobe_height*width] + dImg[index-lobe_height-lobe_height*width];
		Dxy -= tmp;
		
		//assign value to matrix for determinant of Hessian
		Dxy *= Dxy;
		//equation 3 from Bay, et al., 2008
		dDetHess[index+lobe_height]=Dxx*Dyy-.81*Dxy;
	
	}

}
int main(int argc, char** argv)
{
	if(2 >= argc) {
		usage();
		exit(-1);
	}
	char *infilename = argv[1];
	char *templatename = argv[2];
	char outfilename[80];
	sprintf(outfilename,"out_%s",infilename);

	unsigned char *raw_image=NULL;
	unsigned char *temp=NULL;

	int raw_width=0, raw_height=0,
		temp_width=0, temp_height=0;

	/* Try opening a jpeg*/
	if(DEBUG) printf("reading file: %s\n",infilename);
	if( read_jpeg_file(&raw_image, infilename, raw_width, raw_height ) < 0 ) return -1;
	if(DEBUG) printf("reading file: %s\n",templatename);
	if( read_jpeg_file(&temp, templatename, temp_width, temp_height ) < 0 ) return -1;

	/* convert rgb image to gray scale */
	if(DEBUG) printf("converting rgb images to gray scale.\n");
	double * img1 = NULL, *img2 = NULL;
	img1 = (double*) malloc(raw_width*raw_height*sizeof(double)); 
	img2 = (double*) malloc(temp_width*temp_height*sizeof(double)); 
	rgb2gray(raw_image, img1, raw_width, raw_height);	
	rgb2gray(temp, img2, temp_width, temp_height);	

	/* convert images to integral images */
	if(DEBUG) printf("converting images to integral images.\n");
	convert_to_integral(img1, raw_width, raw_height);
	convert_to_integral(img2, temp_width, temp_height);

	/* copy images to gpu */
	if(DEBUG) printf("sending images to gpu.\n");
	double* dImg1 = NULL;
	double* dImg2 = NULL;
	if( cudaMalloc(&dImg1,raw_width*raw_height*sizeof(double)) != cudaSuccess ) return -1;
	if( cudaMalloc(&dImg2,temp_width*temp_height*sizeof(double)) != cudaSuccess ) return -1;
	if( cudaMemcpy(dImg1, img1, raw_width*raw_height*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ) return -1;
	if( cudaMemcpy(dImg2, img2, temp_width*temp_height*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ) return -1;
	
	/* allocate results array for Determinant of Hessian */
	double * dDetHess1 = NULL;
	if( cudaMalloc(&dDetHess1,raw_width*raw_height*sizeof(double)) != cudaSuccess ) return -1;
	double * dDetHess2 = NULL;
	if( cudaMalloc(&dDetHess2,raw_width*raw_height*sizeof(double)) != cudaSuccess ) return -1;
	
	/****** IMAGE PROCESSING *******/
	if(DEBUG) printf("processing images on gpu.\n");
	dim3 threads_per_block(32,32); 
	dim3 num_blocks_img1(ceil(raw_height/threads_per_block.x),ceil(raw_width/threads_per_block.y)); 
	dim3 num_blocks_img2(ceil(temp_height/threads_per_block.x),ceil(temp_width/threads_per_block.y)); 
	int octave = 0;
	calc_det_hessian<<<num_blocks_img1,threads_per_block>>>(dImg1, dDetHess1, raw_width, raw_height);

	/****** END IMAGE PROCESSING *******/

	/* copy results back from card */
	if(DEBUG) printf("retrieving results from gpu.\n");
	int num_bytes = sizeof(double)*raw_width*raw_height;
	double *det_hess = (double*)malloc(num_bytes);
	if(cudaMemcpy(det_hess, dDetHess1, num_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ) return -1;
	
	/* then copy it to another file */
	if(DEBUG) printf("write to file: %s\n",outfilename);
	if( write_jpeg_file(raw_image, outfilename, raw_width, raw_height ) < 0 ) return -1;
	
	/* clean up memory */
	if(DEBUG)printf("cleaning up...\n");

	cudaFree(dImg1);
	dImg1 = NULL;
	cudaFree(dImg2);
	dImg2=NULL;
	if(raw_image) free(raw_image); 
	raw_image=NULL;
	if(temp) free(temp); 
	temp=NULL;

	return 0;
}



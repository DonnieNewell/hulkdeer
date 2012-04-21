#include <stdio.h>
#include <iostream>
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

void rgb2gray(const unsigned char *src, int*dst, size_t width, size_t height){
	for(int i = 0; i<height; i++){
		for(int j = 0; j < width; j++){
			int index = (i*width + j);
				//grayscale
				dst[index] = .30*src[index*bytes_per_pixel] + .59*src[index*bytes_per_pixel+1] + .11*src[index*bytes_per_pixel+2]; 
		}
	}	
}

void convert_to_integral(int *src, size_t width, size_t height){
	for(int i=0; i<height; i++){
		//std::cout <<"\ni:"<<i;
		int west = 0;
		for(int j =0; j < width; j++){
			int index = i*width + j;
			//std::cout << " j:"<<j<<" current:"<<src[index];
			int south = 0;
			
			if(i > 0) 
				south = src[index-width];
			
			int tmp = west+south;
			
			west += src[index];

			src[index] += tmp;
			
			if(0 > src[index]){
				fprintf(stderr, "ERROR: overflow in integral image calculation.\n");
				exit(1);
			}
			
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
	int location = 0;
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
	//int index = (i*width+j);
	if(i < height && j < width){
		int mi = i;
		int mj = j;
		int ms = 0;

		/* find maximum in current grid */
		for(int s = 0; s<4; ++s){
			for(int i2 = i; i2 <= i+n; i2++){
				for(int j2 = j; j2 <= j+n; j2++){
					if(src[s*width*height + i2*width+j2] > src[s*width*height+mi*width+mj]){
						mi = i2;
						mj = j2;
						ms = s;
					}//end if
				}//end for j2
			} //end for i2
		} //end for s

		/* check neighborhood around maximum */
		for(int s = 0; s<4; ++s){
			for(int i2 = mi-n; i2 <= mi+n; i2++){
				for(int j2 = mj-n; j2 <= mj+n; j2++){
					if(src[s*width*height + i2*width+j2] > src[ms*width*height + mi*width+mj]) 
						return;
				}
			}
		}

		/*found a local maximum */
		points[ms*width*height+mi*width + mj] = 1;
	}//end if

}//end nms()

void CPU_calc_det_hessian(int* dImg, double* dDetHess, int width, int height, int interval, int filter_width){
	//calc index
	int half_filter_width = filter_width/2;

	for(int i = half_filter_width; i<height-half_filter_width; ++i){
		for(int j = half_filter_width; j<width-half_filter_width; ++j){

			int lobe_height = filter_width/3;
			int lobe_width = 2*lobe_height-1;


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
			if(0 > Dyy) Dyy=0.0;

			//normalize by filter area
			Dyy /= filter_width*filter_width;


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
			if(0 > Dxx) Dxx=0.0;

			//normalize by filter area
			Dxx /= filter_width*filter_width;


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
			if(0>Dxy) Dxy=0.0;

			//normalize by filter area
			Dxy /= filter_width*filter_width;

			//assign value to matrix for determinant of Hessian
			Dxy *= Dxy;

			int sign_of_laplacian = 1;
			if(0>Dxx+Dyy)sign_of_laplacian = -1;

			//equation 3 from Bay, et al., 2008
			index = interval*width*height + index + lobe_height;
			dDetHess[index]=Dxx*Dyy-.81*Dxy;

			if(0 > dDetHess[index]) dDetHess[index] = 0.0;
			else dDetHess[index] *= sign_of_laplacian;

		}
	}

}

__global__ void calc_det_hessian(int* dImg, double* dDetHess, int width, int height, int interval, int filter_width){
	//calc index
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int half_filter_width = filter_width/2;

	//check if entire filter would fit at this location
	if(i < (height-half_filter_width) && 
			i >= half_filter_width && 
			j < (width-half_filter_width) && 
			j >= half_filter_width){

		int lobe_height = filter_width/3;
		int lobe_width = 2*lobe_height-1;


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
		if(0 > Dyy) Dyy=0.0;

		//normalize by filter area
		Dyy /= filter_width*filter_width;


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
		if(0 > Dxx) Dxx=0.0;

		//normalize by filter area
		Dxx /= filter_width*filter_width;


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
		if(0>Dxy) Dxy=0.0;

		//normalize by filter area
		Dxy /= filter_width*filter_width;
		
		//assign value to matrix for determinant of Hessian
		Dxy *= Dxy;

		int sign_of_laplacian = 1;
		if(0>Dxx+Dyy)sign_of_laplacian = -1;

		//equation 3 from Bay, et al., 2008
		index = interval*width*height + index + lobe_height;
		dDetHess[index]=Dxx*Dyy-.81*Dxy;

		if(0 > dDetHess[index]) dDetHess[index] = 0.0;
		else dDetHess[index] *= sign_of_laplacian;
	
	}

}
void write_hessian(unsigned char* dst, double* det_hess, double* det_hess_cpu, int width, int height){
	double epsilon = .01;	
	for(int i = 0; i<height; i++){
		for(int j = 0; j<width; j++){
			int index = i*width+j;
			double difference = det_hess[index]-det_hess_cpu[index];
			if(det_hess_cpu[index]*det_hess_cpu[index]>1.0){
				//apply det_hess in dst
				dst[index*3] = 255;
				dst[index*3+2] = 0;
				dst[index*3+1] = 0;
			}
		}

	}

}

void mark_points(unsigned char* dst, int* src, int width, int height){
	for(int i = 0; i<height; i++){
		for(int j = 0; j<width; j++){
			int index = i*width+j;
			if(0!=src[index]){
				//mark interest point in dst
				for(int p = i-1; p < i +1; p++){
					for(int q = j-1; q < j +1; q++){
						int tmp = p*width+q;
						dst[tmp*3+1] = 0;
						dst[tmp*3+2] = 0;
						dst[tmp*3] = 255;
					}
				}
			}
		}

	}
}//end mark_points

void output(double *array, const int w, const int h){
	printf("printing:\n");
	for(int i = 0; i<h; i++){
		printf("row %d: ",i);
		for(int j=0; j<w; j++){
			printf("%.2f ",array[i*w+j]);
		}
		printf("\n");
	}
}

void output(int *array, const int w, const int h){
	printf("printing:\n");
	for(int i = 0; i<h; i++){
		printf("row %d: ",i);
		for(int j=0; j<w; j++){
			printf("%d ",array[i*w+j]);
		}
		printf("\n");
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
	int* img1 = NULL, *img2 = NULL;
	img1 = (int*) malloc(raw_width*raw_height*sizeof(int)); 
	img2 = (int*) malloc(temp_width*temp_height*sizeof(int)); 
	rgb2gray(raw_image, img1, raw_width, raw_height);	
	rgb2gray(temp, img2, temp_width, temp_height);	

	/* convert images to integral images */
	if(DEBUG) printf("converting images to integral images.\n");
	convert_to_integral(img1, raw_width, raw_height);
	convert_to_integral(img2, temp_width, temp_height);
	if(DEBUG) {
		int limit = 10;
		int w = raw_width;
		int h = raw_height;
		int *img = img1;
		printf("printing last %d elements in integral image.\n",limit);
		for(int i=height-limit; i<height; i++){
			printf("%d:  ",i);
			for(int j=width-limit; j<width; j++){
				printf("%d ",img[i*w+j]);
			}
			printf("\n");
		}
	}
	/* copy images to gpu */
	if(DEBUG) printf("sending images to gpu.\n");
	int* dImg1 = NULL;
	int* dImg2 = NULL;
	if( cudaMalloc(&dImg1,raw_width*raw_height*sizeof(int)) != cudaSuccess ) return -1;
	if( cudaMalloc(&dImg2,temp_width*temp_height*sizeof(int)) != cudaSuccess ) return -1;
	if( cudaMemcpy(dImg1, img1, raw_width*raw_height*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ) return -1;
	if( cudaMemcpy(dImg2, img2, temp_width*temp_height*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ) return -1;
	
	int num_filters = 4;
	
	/************************************ create cpu array for calculating hessian to compare */
	double * hDetHess1 = NULL;
	hDetHess1 = (double*)malloc(num_filters*raw_width*raw_height*sizeof(double));

	/* allocate results array for Determinant of Hessian */
	double * dDetHess1 = NULL;
	if( cudaMalloc(&dDetHess1,num_filters*raw_width*raw_height*sizeof(double)) != cudaSuccess ) return -1;
	double * dDetHess2 = NULL;
	if( cudaMalloc(&dDetHess2,num_filters*raw_width*raw_height*sizeof(double)) != cudaSuccess ) return -1;
	
	/****** IMAGE PROCESSING *******/
	if(DEBUG) printf("processing images on gpu.\n");
	dim3 threads_per_block(32,32); 
	dim3 num_blocks_img1(ceil(raw_height/(float)threads_per_block.x),ceil(raw_width/(float)threads_per_block.y)); 
	dim3 num_blocks_img2(ceil(temp_height/(float)(threads_per_block.x)),ceil(temp_width/(float)(threads_per_block.y))); 
	//int octave = 0;
	int filter_width = 9;
	int filter_increase = 6;
	int interval = 0;
	for(int i = 0; i<4; ++i){
		CPU_calc_det_hessian(img1, hDetHess1, raw_width, raw_height, i, filter_width+i*filter_increase);
		calc_det_hessian<<<num_blocks_img1,threads_per_block>>>(dImg1, dDetHess1, raw_width, raw_height, i, filter_width+i*filter_increase);
	}

	/* performing non-maximal suppression */
	if(DEBUG) printf("performing non-maximal suppression.\n");
	int* intPoints1 = NULL;
	if( cudaMalloc(&intPoints1,num_filters*raw_width*raw_height*sizeof(int)) != cudaSuccess ) return -1;
	if( cudaMemset(intPoints1,0,num_filters*raw_width*raw_height*sizeof(int)) != cudaSuccess ) return -1;
	int * intPoints2 = NULL;
	if( cudaMalloc(&intPoints2,num_filters*temp_width*temp_height*sizeof(int)) != cudaSuccess ) return -1;
	if( cudaMemset(intPoints2,0,num_filters*temp_width*temp_height*sizeof(int)) != cudaSuccess ) return -1;
	num_blocks_img1.x=ceil(raw_height/(float)(3*threads_per_block.x));
	num_blocks_img1.y=ceil(raw_width/(float)(3*threads_per_block.y)); 
	num_blocks_img2.x=ceil(temp_height/(float)(3*threads_per_block.x));
	num_blocks_img2.y=ceil(temp_width/(float)(3*threads_per_block.y)); 
	non_maximal_suppression<<<num_blocks_img1, threads_per_block>>>(dDetHess1, intPoints1, raw_width, raw_height);

	/****** END IMAGE PROCESSING *******/

	/* copy results back from card */
	if(DEBUG) printf("retrieving results from gpu.\n");
	int num_bytes = num_filters*sizeof(double)*raw_width*raw_height;
	double *det_hess = (double*)malloc(num_bytes);
	if(cudaMemcpy(det_hess, dDetHess1, num_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ) return -1;
		
	num_bytes = num_filters*sizeof(int)*raw_width*raw_height;
	int *interest_points = (int*)malloc(num_bytes);
	if(cudaMemcpy(interest_points, intPoints1, num_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ) return -1;
	
	/* Mark up image */
	if(DEBUG) printf("marking interest points on image\n");
	//mark_points(raw_image, interest_points, raw_width, raw_height);
	write_hessian(raw_image, det_hess,hDetHess1, raw_width, raw_height);

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



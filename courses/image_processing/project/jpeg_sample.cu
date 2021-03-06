#include <stdio.h>
#include <iostream>
#include <jpeglib.h>
#include <stdlib.h>
#include <math.h>
#include <npp.h>

#define DEBUG 1
#define PI 3.14159

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

/* code adapted from http://rosettacode.org/wiki/Bitmap/Bresenham's_line_algorithm#C */
void line(unsigned char* img, int x0, int y0, int x1, int y1, int width, int height) {
 
  int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
  int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1; 
  int err = (dx>dy ? dx : -dy)/2, e2;
  x0 = min(x0,height-1); 
  y0 = min(y0,width-1); 
  x1 = min(x1,height-1); 
  y1 = min(y1,width-1); 
  for(;;){
    img[3*(x0*width+y0)]=255;
    img[3*(x0*width+y0)+1]=0;
    img[3*(x0*width+y0)+2]=0;
    if (x0==x1 && y0==y1) break;
    e2 = err;
    if (e2 >-dx) {
 	err -= dy; 
	x0 += sx; 
	if(x0 >= height) break;
    }
    if (e2 < dy) { 
	err += dx; 
	y0 += sy; 
	if(y0 >= width) break;
    }
  }
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
		//DEBUG
		//points[i*width+j] = 1;	
		//return;

		/* find maximum in current grid */
		for(int s = 0; s<4; ++s){
			for(int i2 = i; i2 <= i+n; i2++){
				for(int j2 = j; j2 <= j+n; j2++){
					if(src[s*width*height + i2*width+j2] * 
						src[s*width*height + i2*width+j2] > 
						src[ms*width*height+mi*width+mj] * 
						src[ms*width*height+mi*width+mj]){
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
					if(src[s*width*height + i2*width+j2] * 
						src[s*width*height + i2*width+j2] > 
						src[ms*width*height+mi*width+mj] * 
						src[ms*width*height+mi*width+mj]){
						return;
					}
				}
			}
		}

		/*found a local maximum */
		if(src[ms*width*height+mi*width+mj] * 
			src[ms*width*height+mi*width+mj]>1.0){
			points[ms*width*height+mi*width + mj] = 1;
		}
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
		index = interval*width*height + i*width+j;
		dDetHess[index]=Dxx*Dyy-.81*Dxy;

		if(0 > dDetHess[index]) dDetHess[index] = 0.0;
		else dDetHess[index] *= sign_of_laplacian;
	
	}

}
void write_interest(unsigned char* dst, int* interest, int width, int height){
	double epsilon = .01;	
	for(int s = 0; s<4; s++){
		for(int i = 0; i<height; i++){
			for(int j = 0; j<width; j++){
				int index = i*width+j;
				int offset = s*width*height;
				if(interest[offset+index] > 0){
					//apply interest in dst
					dst[index*3] = 255;
					dst[index*3+2] = 0;
					dst[index*3+1] = 0;
				}
			}

		}
	}

}

void write_hessian(unsigned char* dst, double* det_hess, int width, int height, int scale){
	double epsilon = .01;	
	for(int i = 0; i<height; i++){
		for(int j = 0; j<width; j++){
			int index = i*width+j;
			int offset = scale*width*height;
			if(det_hess[offset+index]*det_hess[offset+index]>1.0){
				//apply det_hess in dst
				dst[index*3] = 255;
				dst[index*3+2] = 0;
				dst[index*3+1] = 0;
			}
		}

	}

}

void mark_orientations(unsigned char* img, int* interest, int* orient,int num_points, int width, int height){
	for(int pt=0; pt < num_points; pt++){
		int i = interest[pt*3];
		int j = interest[pt*3+1];
		int index = i*width+j;
		int orient_i = orient[pt*2];
		int orient_j = orient[pt*2+1];
		if(i <10 || i > height-10 ||j<10 || j>width-10)
			continue;

		//mark interest point in dst
		double line_length = 10.0;
		double scale = 2.0/sqrt(orient_i*orient_i+orient_j*orient_j);

		//check for nan and infinity
		if( isinf(scale) )
			continue;
		else if( isnan(scale) )
			continue;
		

		int scaled_i= ceil(fabs(orient_i*scale));
		int scaled_j = ceil(fabs(orient_j*scale));
		
		if(orient_i < 0)
			scaled_i = floor(orient_i*scale);
		else
			scaled_i = ceil(orient_i*scale);

		if(orient_j < 0)
			scaled_j = floor(orient_j*scale);
		else
			scaled_j = ceil(orient_j*scale);

	//	fprintf(stderr,"line(img,i:%d,j:%d,i+scaled_i:%d,j+scaled_j:%d,orient_i:%d,orient_j:%d,width,height);\n",i,j,scaled_i,scaled_j,orient_i, orient_j);
		line(img,i,j,i+scaled_i, j+scaled_j,width,height);		
	}
}//end mark_orientations


void mark_points(unsigned char* dst, int* src,int num_points, int width, int height){
	for(int pt=0; pt < num_points; pt++){
		int i = src[pt*3];
		int j = src[pt*3+1];
		int scale = src[pt*3+2];
		int scale_offset = scale*width*height;
		int index = i*width+j;
		//mark interest point in dst
		for(int p = max(0,i-scale-1); p < min(i +scale+1,height); p++){
			for(int q = max(0,j-scale-1); q < min(j +scale+1,width); q++){
				int tmp = p*width+q;

				if(0==scale){	
					dst[tmp*3+1] = 0;
					dst[tmp*3+2] = 0;
					dst[tmp*3] = 255;
				}
				if(1==scale){		
					dst[tmp*3+1] = 0;
					dst[tmp*3+2] = 255;
					dst[tmp*3] = 0;
				}
				if(2==scale){	
					dst[tmp*3+1] = 100;
					dst[tmp*3+2] = 0;
					dst[tmp*3] = 100;
				}
				else{
					dst[tmp*3+1] = 0;
					dst[tmp*3+2] = 255;
					dst[tmp*3] = 255;
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

/* this function approximates the Haar wavelet in x or y direction */
int haar(int* img, int i, int j, int i_width, int i_height, float scale, int direction, int for_descriptor){
	int haar_width = -1;
	if(for_descriptor)
		haar_width = 2*scale;
	else
		haar_width = 4*scale;

	int half_width = haar_width/2;
	int black = 0, white=0;

	if(1==direction){ //y direction
		black = img[i*width+j+half_width]-img[(i-half_width)*width+j+half_width]-
			img[i*width+j-half_width]+img[(i-half_width)*width+j-half_width];
		white = img[(i+half_width)*width+j+half_width]-img[(i)*width+j+half_width]-
			img[(i+half_width)*width+j-half_width]+img[(i)*width+j-half_width];
	}else{ //x direction
		black = img[(i+half_width)*width+j]-img[(i-half_width)*width+j]-
			img[(i+half_width)*width+j-half_width]+img[(i-half_width)*width+j-half_width];
		white = img[(i+half_width)*width+j+half_width]-img[(i-half_width)*width+j+half_width]-
			img[(i+half_width)*width+j]+img[(i-half_width)*width+j];
	}
	return -1*black+white;

}

/* returns the value of 2d gaussian centered at the origin */
float gauss(int x, int y, float sigma){
	float sigma_squared = sigma*sigma;
	return ( 1 / sqrt(2*PI*sigma_squared)) * exp(-(x*x+y*y)/(2*sigma_squared));
}

/* returns the correct index for a relative pixel after rotation */
int getIndex(int rel_i, int rel_j, int itrst_i, int itrst_j, float angle_rotate, int img_width, int img_height){
	int new_x = 0.5f+rel_i*cos(angle_rotate)-rel_j*sin(angle_rotate);
	int new_y = 0.5f+rel_i*sin(angle_rotate)+rel_j*cos(angle_rotate);
	int actual_x = new_x + itrst_i;
	int actual_y = new_y + itrst_j;

	if(actual_x < 0 || actual_x >=img_height || actual_y < 0 || actual_y >=img_width)
		return -1;

	return actual_x*img_width+actual_y;
}

float angle_between(int x0, int y0, int x1, int y1){
	float angle = 0.0f;
	float mag0 = sqrtf(x0*x0+y0*y0);
	float mag1 = sqrtf(x1*x1+y1*y1);
	//	fprintf(stderr,"angle_between:mag0:%f, mag1:%f.\n",mag0,mag1);
	angle = acosf((x0*y0+x1*y1)/(mag0*mag1));
	return angle;
}

/* function calculates the 64-feature descriptor for each interest point */
void calculate_descriptor(int *img, int* intPoints,int num_interest_points, int* orient, float**descriptors, int width, int height){
	
	int row=-1,col=-1,itrvl=-1,orient_i=-1,orient_j=-1;
	int filter_size = -1, window_size=-1;
	int step = -1;
	float scale = 0.0f;
	float sigma = 0.0f;
	float *neighborhood = NULL;

	for(int int_pt = 0; int_pt < num_interest_points; ++int_pt){
		orient_i = orient[2*int_pt];
		orient_j = orient[2*int_pt+1];
		//printf("orient_i:%d, orient_j:%d\n",orient_i, orient_j);
		neighborhood = NULL;
		if(orient_i==0 && orient_j==0) 
			continue;

		//if(DEBUG) fprintf(stderr,"calculating descriptor for interest point:%d of %d.\n",int_pt,num_interest_points);
		neighborhood = (float*)malloc(4*8*8*sizeof(float)); //simplifying the neighborhood by not using sub samples
		//each neighborhood has dx, dy, |dx|, and |dy|


		//calculate descriptor for this interest point
		row = intPoints[int_pt*3];
		col = intPoints[int_pt*3+1];
		itrvl = intPoints[int_pt*3+2];
		scale = (9+itrvl*6)/9;
		filter_size=2*scale;
		window_size = 20*scale;
		step = window_size/8; //this is to evenly space the samples in the window
		sigma = 3.3*scale;
		//angle between the orientation vector and vertical
		float angle = angle_between(orient_i,orient_j,0,5);
		
		//if(DEBUG) fprintf(stderr,"calculating Haar responses for the neighborhood.\n");
		//get Haar wavelet responses at each point
		for(int i = 0; i<8; ++i){
			for(int j=0; j<8; ++j){
				//relative points scaled
				int i_sample = (i-4)*step;
				int j_sample = (j-4)*step;
				float weight = gauss(i_sample,j_sample,sigma);
				int index = getIndex(i_sample,j_sample,row,col,angle,width,height);
				
				//if(index < 0)printf("index:%d <-- getIndex(is:%d,js:%d,r:%d,c:%d,a:%f,w:%d,h:%d);\n",index,i_sample,j_sample,row,col,angle,width,height);
				float temp = 0.0f;
				float abs_dx = 0.0f;
				float abs_dy = 0.0f;
				float tempdx=0.0f;
				float tempdy = 0.0f;
				if(index >=0 && 
					index/width > (int)scale && 
					index%width > (int)scale && 
					index/width < (int)(height-scale) && 
					index%width < (int)(width-scale)){
					//down one row
					temp = weight*haar(img,min((index/width)+1,height-1),index%width, width, height, scale,0,1);
					abs_dx = fabs(temp);
					tempdx = temp;
					temp = weight*haar(img,min((index/width)+1,height-1),index%width, width, height, scale,1,1);
					abs_dy = fabs(temp);
					tempdy = temp;

					//up one row
					temp = weight*haar(img,max((index/width)-1,0),index%width, width, height, scale,0,1);
					abs_dx += fabs(temp);
					tempdx += temp;
					temp = weight*haar(img,max((index/width)-1,0),index%width, width, height, scale,1,1);
					abs_dy += fabs(temp);
					tempdy += temp;

					//right one column
					temp = weight*haar(img,index/width,min((index%width)+1,width-1), width, height, scale,0,1);
					abs_dx += fabs(temp);
					tempdx += temp;
					temp = weight*haar(img,index/width,min((index%width)+1,width-1), width, height, scale,1,1);
					abs_dy += fabs(temp);
					tempdy += temp;

					//left one column
					temp = weight*haar(img,index/width,max((index%width)-1,0), width, height, scale,0,1);
					abs_dx += fabs(temp);
					tempdx += temp;
					temp = weight*haar(img,index/width,max((index%width)-1,0), width, height, scale,1,1);
					abs_dy += fabs(temp);
					tempdy += temp;

					float mag_abs = sqrt(abs_dx*abs_dx+abs_dy*abs_dy);				
					float mag_temp = sqrt(tempdx*tempdx+tempdy*tempdy);				
					float a2 = angle_between(tempdx,tempdy,orient_i,orient_j);
					float a3 = angle_between(abs_dx,abs_dy,orient_i,orient_j);
					
					//convert haar vector to be relative to the orientation vector of the interest point.
					abs_dx = mag_abs*cosf(a3);
					abs_dy = mag_abs*sinf(a3);
					tempdx = mag_temp*cosf(a2);
					tempdy = mag_temp*sinf(a2);
//					if(abs_dx==0.0f || abs_dy==0.0f || tempdx==0.0f || tempdy==0.0f)
//						fprintf(stderr,"int_pt:%d -- abs_dx:%f, abs_dy:%f, tempdx:%f, tempdy:%f\n",int_pt,abs_dx,abs_dy,tempdx,tempdy);

				}
				neighborhood[4*(i*8+j)]= tempdx;
				neighborhood[4*(i*8+j)+1]= abs_dx;
				neighborhood[4*(i*8+j)+2]= tempdy;
				neighborhood[4*(i*8+j)+3]= abs_dy;

			}//end for cols
		}//end for rows

		//we now have the haar
		descriptors[int_pt]=neighborhood;
		neighborhood = NULL;

	}//end for interest points
}//end calc descriptor


/* this function calculates the 'orientation' of  each interest point, using Haar wavelets */
void calculate_orientation(int *img, int* intPoints,int num_interest_points, int* orient, int width, int height){
	int row=-1,col=-1,itrvl=-1;
	float scale = 0.0f;
	int radius=0;
	float sigma=0;
	float*neighborhood = NULL;
	neighborhood = (float*)malloc(2*17424*sizeof(float)); //size of neighborhood if scale is 11

	for(int curr = 0; curr < num_interest_points; ++curr){
		//get current interest point
		row = intPoints[curr*3];
		col = intPoints[curr*3+1];
		itrvl = intPoints[curr*3+2];
		scale = (9+itrvl*6)/9;
		radius = 6*scale;
		sigma = 2*scale;


		//only calculate if there's enough room
		if(row < radius || row > height-radius || 
				col < radius || col > width-radius) continue;

		//calculate weighted Haar response values for neighborhood around interest point		
		int n_width = 2*radius;
		int in_circle = 0;
		for(int i= row-radius; i<row+radius; i++){
			for(int j=col-radius; j<col+radius; j++){
				int n_row = (i-row+radius);
				int n_col = j-col+radius;
				int delta_row = i-row;
				int delta_col = j-col;
				delta_row*= delta_row;
				delta_col*= delta_col;
				in_circle = sqrt(delta_row+delta_col) <= radius;
				if(in_circle){
					//printf("in the circle***************************************\n");
					float weight = gauss(delta_row,delta_col,sigma);
					int tmp_index = 2*(n_row*n_width+n_col);
					neighborhood[tmp_index] = weight*haar(img,i,j,width,height,scale,0, 0);
					neighborhood[tmp_index+1] = weight*haar(img,i,j,width,height,scale,1, 0);
					//printf("neighb[%d]:%f, neighb[%d+1]:%f.\n",tmp_index,neighborhood[tmp_index],tmp_index,neighborhood[tmp_index]);
				}else{
					neighborhood[2*(n_row*n_width+n_col)]=0.0f;
					neighborhood[2*(n_row*n_width+n_col)+1]=0.0f;
				}
			}
		}

		//determine 'orientation' of interest point
		float step = PI/6.0f;
		float range = PI/3.0f;
		double max_x=0.0,max_y=0.0;
		for(float start=0.0f; start<5*PI/3;start+=step){
			float stop =  start+range;
			//now go through all points in neighborhood and sum if they are in this window.
			double sum_x=0.0,sum_y=0.0;
			for(int i=0; i<2*radius; i++){
				for(int j=0; j<2*radius; j++){
					//haar in y divided by haar in x is the slope to check
					int index = i*2*radius+j;
					float angle = angle_between(neighborhood[2*index],neighborhood[2*index+1],0,5);
					if(angle >=start && angle < stop){
						//point is in the sliding window
						//printf("points are being added****************\n");
						sum_x+=neighborhood[2*index];
						sum_y+=neighborhood[2*index+1];
					}
				}
			
			}

			double length_of_max = sqrt(max_x*max_x+max_y*max_y);
			double curr_length = sqrt(sum_x*sum_x+sum_y*sum_y);
			if(curr_length > length_of_max){
				max_x = sum_x;
				max_y = sum_y;
			}
		}//end for

		//now set the orientation vector
		orient[curr*2]=max_x;
		orient[curr*2+1]=max_y;
	}//end for all interest points

	//clean up
	free(neighborhood);
	
}

int count_interest_points(int *interest_points, int width, int height){
	int num_points =0;
	for(int s = 0; s < 4; s++){
		for(int i=0; i < height; i++){
			for(int j=0; j<width; j++){
				if(interest_points[i*width+j])
					num_points++;
			}
		}
	}
	return num_points;
}	

void get_compact_interest_pts(int *interest_points, int *compact_interest_pts, int width, int height){
	int curr_point = 0;
	for(int s = 0; s < 4; s++){
		for(int i=0; i < height; i++){
			for(int j=0; j<width; j++){
				if(interest_points[i*width+j]){
					compact_interest_pts[curr_point++]=i;
					compact_interest_pts[curr_point++]=j;
					compact_interest_pts[curr_point++]=s;
				}
			}
		}
	}

}

bool match_images(float**descriptors1, int num_desc1, float**descriptors2, int num_desc2, float thresh){
	int num_non_null = 0;
	int num_match = 0;
	printf("image1 num_desc1:%d, image2 num_desc2:%d.\n",num_desc1,num_desc2);
	for(int i = 0; i<min(num_desc1,num_desc2); i++){
		if(descriptors1[i] != NULL && descriptors2[i] != NULL){
			num_non_null++;
			int match = 1;
			//check all of the descriptors
			for(int j=0; j<64; j++){
				if(descriptors1[i][j*4] != descriptors2[i][j*4]  ){
					match = 0;
					break;
				}
				if(descriptors1[i][j*4+1] != descriptors2[i][j*4+1]){
					match = 0;
					break;
				}
				if(descriptors1[i][j*4+2] != descriptors2[i][j*4+2]){
					match = 0;
					break;
				}
				if(descriptors1[i][j*4+3] != descriptors2[i][j*4+3]){
					match = 0;
					break;
				}

			}
			if(match)
				num_match++;
		}
	}
	printf("num matched: %d, num_non_null: %d\n",num_match,num_non_null);
	if(thresh <= num_match/(float)num_non_null)
		return true;
	else
		return false;
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
	char outfilename2[80];
	sprintf(outfilename2,"out_%s",templatename);

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
	if(0) {
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
	int* dImg1 = NULL;
	int* dImg2 = NULL;
	if(DEBUG) printf("allocate space on gpu.\n");
	if( cudaMalloc(&dImg1,raw_width*raw_height*sizeof(int)) != cudaSuccess ) return -1;
	if( cudaMalloc(&dImg2,temp_width*temp_height*sizeof(int)) != cudaSuccess ) return -1;
	if(DEBUG) printf("copy images to gpu.\n");
	if( cudaMemcpy(dImg1, img1, raw_width*raw_height*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ) return -1;
	if( cudaMemcpy(dImg2, img2, temp_width*temp_height*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ) return -1;
	
	int num_filters = 4;
	
	/************************************ create cpu array for calculating hessian to compare */
	double * hDetHess1 = NULL;
	hDetHess1 = (double*)malloc(num_filters*raw_width*raw_height*sizeof(double));
	double * hDetHess2 = NULL;
	hDetHess2 = (double*)malloc(num_filters*temp_width*temp_height*sizeof(double));

	/* allocate results array for Determinant of Hessian */
	double * dDetHess1 = NULL;
	if( cudaMalloc(&dDetHess1,num_filters*raw_width*raw_height*sizeof(double)) != cudaSuccess ) return -1;
	double * dDetHess2 = NULL;
	if( cudaMalloc(&dDetHess2,num_filters*temp_width*temp_height*sizeof(double)) != cudaSuccess ) return -1;
	
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
//		CPU_calc_det_hessian(img1, hDetHess1, raw_width, raw_height, i, filter_width+i*filter_increase);
		calc_det_hessian<<<num_blocks_img1,threads_per_block>>>(dImg1, dDetHess1, raw_width, raw_height, i, filter_width+i*filter_increase);
		calc_det_hessian<<<num_blocks_img2,threads_per_block>>>(dImg2, dDetHess2, temp_width, temp_height, i, filter_width+i*filter_increase);
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
	non_maximal_suppression<<<num_blocks_img2, threads_per_block>>>(dDetHess2, intPoints2, temp_width, temp_height);


	/* copy interest points back to host */
	if(DEBUG) printf("copying interest points back to host.\n");
	int num_bytes1 = num_filters*sizeof(int)*raw_width*raw_height;
	int num_bytes2 = num_filters*sizeof(int)*temp_width*temp_height;
	int *interest_points1 = (int*)malloc(num_bytes1);
	int *interest_points2 = (int*)malloc(num_bytes2);
	if(cudaMemcpy(interest_points1, intPoints1, num_bytes1, cudaMemcpyDeviceToHost) != cudaSuccess ) return -1;
	if(cudaMemcpy(interest_points2, intPoints2, num_bytes2, cudaMemcpyDeviceToHost) != cudaSuccess ) return -1;
	int num_interest_points1 = count_interest_points(interest_points1,raw_width, raw_height);	
	int num_interest_points2 = count_interest_points(interest_points2,temp_width, temp_height);	
	num_bytes1 = 3*num_interest_points1*sizeof(int);
	num_bytes2 = 3*num_interest_points2*sizeof(int);
	int *compact_interest_pts1 = (int*)malloc(num_bytes1);
	int *compact_interest_pts2 = (int*)malloc(num_bytes2);
	get_compact_interest_pts(interest_points1, compact_interest_pts1, raw_width, raw_height);
	get_compact_interest_pts(interest_points2, compact_interest_pts2, temp_width, temp_height);
	
	/* clean up memory */
	if(interest_points1){
	 	free(interest_points1);
		interest_points1=NULL;
	}
	if(interest_points2){
	 	free(interest_points2);
		interest_points2=NULL;
	}
		

	/* get orientation of interest points */
	if(DEBUG) printf("calculating orientation of interest points.\n");
	int * orient1= NULL;
	int * orient2= NULL;
	orient1 = (int *)malloc(2*num_interest_points1*sizeof(int));
	orient2 = (int *)malloc(2*num_interest_points2*sizeof(int));
	calculate_orientation(img1, compact_interest_pts1, num_interest_points1, orient1, raw_width, raw_height);
	calculate_orientation(img2, compact_interest_pts2, num_interest_points2, orient2, temp_width, raw_height);

	/* calculate descriptors for each interest point */
	if(DEBUG) printf("calculating descriptors for interest points.\n");
	float** descriptors1 = NULL;
	float** descriptors2 = NULL;
	descriptors1 = (float**)malloc(num_interest_points1*sizeof(float*));
	descriptors2 = (float**)malloc(num_interest_points2*sizeof(float*));
	bzero(descriptors1,num_interest_points1*sizeof(float*));
	bzero(descriptors2,num_interest_points2*sizeof(float*));
	calculate_descriptor(img1, compact_interest_pts1, num_interest_points1, orient1, descriptors1, raw_width, raw_height);
	calculate_descriptor(img2, compact_interest_pts2, num_interest_points2, orient2, descriptors2, temp_width, temp_height);
	
	/* checking if descriptors match for interest points in the images */
	if(DEBUG) printf("comparing descriptors between image interest points.\n");
	if(match_images(descriptors1,num_interest_points1,descriptors2,num_interest_points2,0.1))
		printf("********************IMAGES HAVE MATCHING INTEREST POINTS*****************************\n");
	else
		printf("-----------------DON'T MATCH------------------------------\n");
	/****** END IMAGE PROCESSING *******/
	
	/* copy results back from card */
	if(DEBUG) printf("retrieving results from gpu.\n");
	num_bytes1 = num_filters*sizeof(double)*raw_width*raw_height;
	num_bytes2 = num_filters*sizeof(double)*temp_width*temp_height;
	double *det_hess1 = (double*)malloc(num_bytes1);
	double *det_hess2 = (double*)malloc(num_bytes2);
	if(cudaMemcpy(det_hess1, dDetHess1, num_bytes1, cudaMemcpyDeviceToHost) != cudaSuccess ) return -1;
	if(cudaMemcpy(det_hess2, dDetHess2, num_bytes2, cudaMemcpyDeviceToHost) != cudaSuccess ) return -2;
		

	/* Mark up image */
	//if(DEBUG) printf("marking interest points on image\n");
	mark_orientations(raw_image, compact_interest_pts1, orient1, num_interest_points1, raw_width, raw_height);
	mark_orientations(temp, compact_interest_pts2, orient2, num_interest_points2, temp_width, temp_height);
	int scale = 3;
	//write_interest(raw_image, interest_points, raw_width, raw_height);
	//write_hessian(raw_image, det_hess1, raw_width, raw_height, scale);

	/* then copy it to another file */
	if(DEBUG) printf("write to file: %s\n",outfilename);
	if( write_jpeg_file(raw_image, outfilename, raw_width, raw_height ) < 0 ) return -1;
	if(DEBUG) printf("write to file: %s\n",outfilename2);
	if( write_jpeg_file(temp, outfilename2, temp_width, temp_height ) < 0 ) return -1;
	
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
	if(descriptors1){
		for(int i=0;i<num_interest_points1;i++){
			if(descriptors1[i]!=NULL) 
				free(descriptors1[i]);
		}
		free(descriptors1);
	}

	return 0;
}



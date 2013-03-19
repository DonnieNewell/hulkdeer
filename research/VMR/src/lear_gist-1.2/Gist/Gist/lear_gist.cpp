/* Lear's GIST implementation, version 1.1, (c) INRIA 2009, Licence: PSFL */
/*--------------------------------------------------------------------------*/

#ifdef USE_GIST

/*--------------------------------------------------------------------------*/

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <fftw3.h>
#include <pthread.h>

#ifdef STANDALONE_GIST

#include "lear_gist.h"

#else 

#include <image.h>
#include <descriptors.h>

#endif

/*--------------------------------------------------------------------------*/

static pthread_mutex_t fftw_mutex = PTHREAD_MUTEX_INITIALIZER;

static void fftw_lock(void)
{
  pthread_mutex_lock(&fftw_mutex);
}

static void fftw_unlock(void)
{
  pthread_mutex_unlock(&fftw_mutex);
}

/*--------------------------------------------------------------------------*/

static image_t *image_add_padding(image_t *src, int padding)
{
    int i, j;

    image_t *img = image_new(src->width + 2*padding, src->height + 2*padding);

    memset(img->data, 0x00, img->stride*img->height*sizeof(float));

    for(j = 0; j < src->height; j++)
    {
        for(i = 0; i < src->width; i++) {
            img->data[(j+padding)*img->stride+i+padding] = src->data[j*src->stride+i];
        }
    }

    for(j = 0; j < padding; j++)
    {
        for(i = 0; i < src->width; i++)
        {
            img->data[j*img->stride+i+padding] = src->data[(padding-j-1)*src->stride+i];
            img->data[(j+padding+src->height)*img->stride+i+padding] = src->data[(src->height-j-1)*src->stride+i];
        }
    }

    for(j = 0; j < img->height; j++)
    {
        for(i = 0; i < padding; i++)
        {
            img->data[j*img->stride+i] = img->data[j*img->stride+padding+padding-i-1];
            img->data[j*img->stride+i+padding+src->width] = img->data[j*img->stride+img->width-padding-i-1];
        }
    }

    return img;
}

/*--------------------------------------------------------------------------*/

static color_image_t *color_image_add_padding(color_image_t *src, int padding)
{
    int i, j;

    color_image_t *img = color_image_new(src->width + 2*padding, src->height + 2*padding);

    for(j = 0; j < src->height; j++)
    {
        for(i = 0; i < src->width; i++)
        {
            img->c1[(j+padding)*img->width+i+padding] = src->c1[j*src->width+i];
            img->c2[(j+padding)*img->width+i+padding] = src->c2[j*src->width+i];
            img->c3[(j+padding)*img->width+i+padding] = src->c3[j*src->width+i];
        }
    }

    for(j = 0; j < padding; j++)
    {
        for(i = 0; i < src->width; i++)
        {
            img->c1[j*img->width+i+padding] = src->c1[(padding-j-1)*src->width+i];
            img->c2[j*img->width+i+padding] = src->c2[(padding-j-1)*src->width+i];
            img->c3[j*img->width+i+padding] = src->c3[(padding-j-1)*src->width+i];

            img->c1[(j+padding+src->height)*img->width+i+padding] = src->c1[(src->height-j-1)*src->width+i];
            img->c2[(j+padding+src->height)*img->width+i+padding] = src->c2[(src->height-j-1)*src->width+i];
            img->c3[(j+padding+src->height)*img->width+i+padding] = src->c3[(src->height-j-1)*src->width+i];
        }
    }

    for(j = 0; j < img->height; j++)
    {
        for(i = 0; i < padding; i++)
        {
            img->c1[j*img->width+i] = img->c1[j*img->width+padding+padding-i-1];
            img->c2[j*img->width+i] = img->c2[j*img->width+padding+padding-i-1];
            img->c3[j*img->width+i] = img->c3[j*img->width+padding+padding-i-1];

            img->c1[j*img->width+i+padding+src->width] = img->c1[j*img->width+img->width-padding-i-1];
            img->c2[j*img->width+i+padding+src->width] = img->c2[j*img->width+img->width-padding-i-1];
            img->c3[j*img->width+i+padding+src->width] = img->c3[j*img->width+img->width-padding-i-1];
        }
    }

    return img;
}

/*--------------------------------------------------------------------------*/

static void image_rem_padding(image_t *dest, image_t *src, int padding)
{
    int i, j;

    for(j = 0; j < dest->height; j++)
    {
        for(i = 0; i < dest->width; i++) {
            dest->data[j*dest->stride+i] = src->data[(j+padding)*src->stride+i+padding];
        }
    }
}

/*--------------------------------------------------------------------------*/

static void color_image_rem_padding(color_image_t *dest, color_image_t *src, int padding)
{
    int i, j;

    for(j = 0; j < dest->height; j++)
    {
        for(i = 0; i < dest->width; i++)
        {
            dest->c1[j*dest->width+i] = src->c1[(j+padding)*src->width+i+padding];
            dest->c2[j*dest->width+i] = src->c2[(j+padding)*src->width+i+padding];
            dest->c3[j*dest->width+i] = src->c3[(j+padding)*src->width+i+padding];
        }
    }
}

/*--------------------------------------------------------------------------*/

static void fftshift(float *data, int w, int h)
{
    int i, j;

    float *buff = (float *) malloc(w*h*sizeof(float));

    memcpy(buff, data, w*h*sizeof(float));

    for(j = 0; j < (h+1)/2; j++)
    {
        for(i = 0; i < (w+1)/2; i++) {
            data[(j+h/2)*w + i+w/2] = buff[j*w + i];
        }

        for(i = 0; i < w/2; i++) {
            data[(j+h/2)*w + i] = buff[j*w + i+(w+1)/2];
        }
    }

    for(j = 0; j < h/2; j++)
    {
        for(i = 0; i < (w+1)/2; i++) {
            data[j*w + i+w/2] = buff[(j+(h+1)/2)*w + i];
        }

        for(i = 0; i < w/2; i++) {
            data[j*w + i] = buff[(j+(h+1)/2)*w + i+(w+1)/2];
        }
    }

    free(buff);
}

/*--------------------------------------------------------------------------*/

static image_list_t *create_gabor(int nscales, const int *or, int width, int height)
{
    int i, j, fn;

    image_list_t *G = image_list_new();
    
    int nfilters = 0;
	float **param = NULL;
    
	float* fx = NULL;
	float *fy = NULL;
	float *fr = NULL;
	float *f = NULL;

	int l = 0;
    
	for(i=0;i<nscales;i++)  nfilters+=or[i];

    param = (float **)malloc(nscales * nfilters * sizeof(float *));
	for(i = 0; i < nscales * nfilters; i++) {
        param[i] = (float *) malloc(4*sizeof(float));
    }

    fx = (float *) malloc(width*height*sizeof(float));
    fy = (float *) malloc(width*height*sizeof(float));
    fr = (float *) malloc(width*height*sizeof(float));
    f  = (float *) malloc(width*height*sizeof(float));

    for(i = 1; i <= nscales; i++)
    {
        for(j = 1; j <= or[i-1]; j++)
        {
            param[l][0] = 0.35f;
            param[l][1] = 0.3f/pow(1.85f, i-1);
            param[l][2] = 16.0f*or[i-1]*or[i-1]/(32*32);
            param[l][3] = M_PI/((float)(or[i-1]))*(j-1);
            l++;
        }
    }

    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            fx[j*width + i] = (float) i - width/2.0f;
            fy[j*width + i] = (float) j - height/2.0f;
            fr[j*width + i] = sqrt(fx[j*width + i]*fx[j*width + i] + fy[j*width + i]*fy[j*width + i]);
            f[j*width + i]  = atan2(fy[j*width + i], fx[j*width + i]);
        }
    }

    fftshift(fr, width, height);
    fftshift(f, width, height);

    for(fn = 0; fn < nfilters; fn++)
    {
        image_t *G0 = image_new(width, height);

        float *f_ptr = f;
        float *fr_ptr = fr;

        for(j = 0; j < height; j++)
        {
            for(i = 0; i < width; i++)
            {
                float tmp = *f_ptr++ + param[fn][3];

                if(tmp < -M_PI) {
                    tmp += 2.0f*M_PI;
                }
                else if (tmp > M_PI) {
                    tmp -= 2.0f*M_PI;
                }

                G0->data[j*G0->stride+i] = exp(-10.0f*param[fn][0]*(*fr_ptr/height/param[fn][1]-1)*(*fr_ptr/width/param[fn][1]-1)-2.0f*param[fn][2]*M_PI*tmp*tmp);
                fr_ptr++;
            }
        }

        image_list_append(G, G0);
    }

    for(i = 0; i < nscales * nfilters; i++) {
        free(param[i]);
    }
    free(param);
	
    free(fx);
    free(fy);
    free(fr);
    free(f);

    return G;
}

/*--------------------------------------------------------------------------*/

/*static*/ void prefilt(image_t *src, int fc)
{
    int i, j;
	
	image_t *img_pad;

	int width = 0;
    int height = 0;
    int stride = 0;

    /* Alloc memory */
    float *fx  = NULL;
    float *fy  = NULL;
    float *gfc = NULL;
    fftwf_complex *in1 = NULL;
    fftwf_complex *in2 = NULL;
    fftwf_complex *out = NULL;
	float s1 = 0.0f;
    
	fftwf_plan fft1;
	fftwf_plan ifft1;
	
	fftwf_plan fft2;
	fftwf_plan ifft2 ;

	fftw_lock();


    /* Log */
    for(j = 0; j < src->height; j++)
    {
        for(i = 0; i < src->width; i++) {
            src->data[j*src->stride+i] = log(src->data[j*src->stride+i]+1.0f);
        }
    }

    img_pad = image_add_padding(src, 5);

    /* Get sizes */
    width = img_pad->width;
    height = img_pad->height;
    stride = img_pad->stride;

    /* Alloc memory */
    fx  = (float *) fftwf_malloc(width*height*sizeof(float));
    fy  = (float *) fftwf_malloc(width*height*sizeof(float));
    gfc = (float *) fftwf_malloc(width*height*sizeof(float));
    in1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    in2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    out = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));

    /* Build whitening filter */
    s1 = fc/sqrt(log(2.0f));
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            in1[j*width + i][0] = img_pad->data[j*stride+i];
            in1[j*width + i][1] = 0.0f;

            fx[j*width + i] = (float) i - width/2.0f;
            fy[j*width + i] = (float) j - height/2.0f;

            gfc[j*width + i] = exp(-(fx[j*width + i]*fx[j*width + i] + fy[j*width + i]*fy[j*width + i]) / (s1*s1));
        }
    }

    fftshift(gfc, width, height);

    /* FFT */
    fft1 = fftwf_plan_dft_2d(width, height, in1, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_unlock();
    fftwf_execute(fft1);
    fftw_lock();

    /* Apply whitening filter */
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            out[j*width+i][0] *= gfc[j*width + i];
            out[j*width+i][1] *= gfc[j*width + i];
        }
    }

    /* IFFT */
    ifft1 = fftwf_plan_dft_2d(width, height, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_unlock();
    fftwf_execute(ifft1);
    fftw_lock();

    /* Local contrast normalisation */
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            img_pad->data[j*stride + i] -= in2[j*width+i][0] / (width*height);

            in1[j*width + i][0] = img_pad->data[j*stride + i] * img_pad->data[j*stride + i];
            in1[j*width + i][1] = 0.0f;
        }
    }

    /* FFT */
    fft2 = fftwf_plan_dft_2d(width, height, in1, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_unlock();
    fftwf_execute(fft2);
    fftw_lock();

    /* Apply contrast normalisation filter */
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            out[j*width+i][0] *= gfc[j*width + i];
            out[j*width+i][1] *= gfc[j*width + i];
        }
    }

    /* IFFT */
    ifft2 = fftwf_plan_dft_2d(width, height, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_unlock();
    fftwf_execute(ifft2);
    fftw_lock();

    /* Get result from contrast normalisation filter */
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++) {
            img_pad->data[j*stride+i] = img_pad->data[j*stride + i] / (0.2f+sqrt(sqrt(in2[j*width+i][0]*in2[j*width+i][0]+in2[j*width+i][1]*in2[j*width+i][1]) / (width*height)));
        }
    }

    image_rem_padding(src, img_pad, 5);

    /* Free */
    fftwf_destroy_plan(fft1);
    fftwf_destroy_plan(fft2);
    fftwf_destroy_plan(ifft1);
    fftwf_destroy_plan(ifft2);

    image_delete(img_pad);

    fftwf_free(in1);
    fftwf_free(in2);
    fftwf_free(out);
    fftwf_free(fx);
    fftwf_free(fy);
    fftwf_free(gfc);

    fftw_unlock();
}

/*--------------------------------------------------------------------------*/

static void color_prefilt(color_image_t *src, int fc)
{
    int i, j;

	color_image_t *img_pad;

	/* Get sizes */
    int width = 0;
    int height = 0;

	
    /* Alloc memory */
    float *fx  = NULL;
    float *fy  = NULL;
    float *gfc = NULL;
    fftwf_complex *ina1 = NULL;
    fftwf_complex *ina2 = NULL;
    fftwf_complex *ina3 = NULL;
    fftwf_complex *inb1 = NULL;
    fftwf_complex *inb2 = NULL;
    fftwf_complex *inb3 = NULL;
    fftwf_complex *out1 = NULL;
    fftwf_complex *out2 = NULL;
    fftwf_complex *out3 = NULL;

	    /* FFT */
    fftwf_plan fft11;
    fftwf_plan fft12;
    fftwf_plan fft13;

	fftwf_plan ifft11;
    fftwf_plan ifft12;
    fftwf_plan ifft13;
   
    /* Build whitening filter */
    float s1 = 0.0f;
    float mean = 0.0f;

	fftwf_plan fft21;
	fftwf_plan ifft2;

	float val = 0.0f;

	fftw_lock();


    /* Log */
    for(j = 0; j < src->height; j++)
    {
        for(i = 0; i < src->width; i++)
        {
            src->c1[j*src->width+i] = log(src->c1[j*src->width+i]+1.0f);
            src->c2[j*src->width+i] = log(src->c2[j*src->width+i]+1.0f);
            src->c3[j*src->width+i] = log(src->c3[j*src->width+i]+1.0f);
        }
    }

    img_pad = color_image_add_padding(src, 5);

    /* Get sizes */
    width = img_pad->width;
    height = img_pad->height;

    /* Alloc memory */
    fx  = (float *) fftwf_malloc(width*height*sizeof(float));
    fy  = (float *) fftwf_malloc(width*height*sizeof(float));
    gfc = (float *) fftwf_malloc(width*height*sizeof(float));
    ina1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    ina2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    ina3 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    inb1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    inb2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    inb3 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    out1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    out2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    out3 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));

    /* Build whitening filter */
    s1 = fc/sqrt(log(2.0f));
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            ina1[j*width + i][0] = img_pad->c1[j*width+i];
            ina2[j*width + i][0] = img_pad->c2[j*width+i];
            ina3[j*width + i][0] = img_pad->c3[j*width+i];
            ina1[j*width + i][1] = 0.0f;
            ina2[j*width + i][1] = 0.0f;
            ina3[j*width + i][1] = 0.0f;

            fx[j*width + i] = (float) i - width/2.0f;
            fy[j*width + i] = (float) j - height/2.0f;

            gfc[j*width + i] = exp(-(fx[j*width + i]*fx[j*width + i] + fy[j*width + i]*fy[j*width + i]) / (s1*s1));
        }
    }

    fftshift(gfc, width, height);

    /* FFT */
    fft11 = fftwf_plan_dft_2d(width, height, ina1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    fft12 = fftwf_plan_dft_2d(width, height, ina2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
    fft13 = fftwf_plan_dft_2d(width, height, ina3, out3, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_unlock();
    fftwf_execute(fft11);
    fftwf_execute(fft12);
    fftwf_execute(fft13);
    fftw_lock();

    /* Apply whitening filter */
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            out1[j*width+i][0] *= gfc[j*width + i];
            out2[j*width+i][0] *= gfc[j*width + i];
            out3[j*width+i][0] *= gfc[j*width + i];

            out1[j*width+i][1] *= gfc[j*width + i];
            out2[j*width+i][1] *= gfc[j*width + i];
            out3[j*width+i][1] *= gfc[j*width + i];
        }
    }

    /* IFFT */
    ifft11 = fftwf_plan_dft_2d(width, height, out1, inb1, FFTW_BACKWARD, FFTW_ESTIMATE);
    ifft12 = fftwf_plan_dft_2d(width, height, out2, inb2, FFTW_BACKWARD, FFTW_ESTIMATE);
    ifft13 = fftwf_plan_dft_2d(width, height, out3, inb3, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_unlock();
    fftwf_execute(ifft11);
    fftwf_execute(ifft12);
    fftwf_execute(ifft13);
    fftw_lock();

    /* Local contrast normalisation */
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            img_pad->c1[j*width+i] -= inb1[j*width+i][0] / (width*height);
            img_pad->c2[j*width+i] -= inb2[j*width+i][0] / (width*height);
            img_pad->c3[j*width+i] -= inb3[j*width+i][0] / (width*height);

            mean = (img_pad->c1[j*width+i] + img_pad->c2[j*width+i] + img_pad->c3[j*width+i])/3.0f;

            ina1[j*width+i][0] = mean*mean;
            ina1[j*width+i][1] = 0.0f;
        }
    }

    /* FFT */
    fft21 = fftwf_plan_dft_2d(width, height, ina1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_unlock();
    fftwf_execute(fft21);
    fftw_lock();

    /* Apply contrast normalisation filter */
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            out1[j*width+i][0] *= gfc[j*width + i];
            out1[j*width+i][1] *= gfc[j*width + i];
        }
    }

    /* IFFT */
    ifft2 = fftwf_plan_dft_2d(width, height, out1, inb1, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_unlock();
    fftwf_execute(ifft2);
    fftw_lock();

    /* Get result from contrast normalisation filter */
    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            val = sqrt(sqrt(inb1[j*width+i][0]*inb1[j*width+i][0]+inb1[j*width+i][1]*inb1[j*width+i][1]) / (width*height));

            img_pad->c1[j*width+i] /= (0.2f+val);
            img_pad->c2[j*width+i] /= (0.2f+val);
            img_pad->c3[j*width+i] /= (0.2f+val);
        }
    }

    color_image_rem_padding(src, img_pad, 5);

    /* Free */
    fftwf_destroy_plan(fft11);
    fftwf_destroy_plan(fft12);
    fftwf_destroy_plan(fft13);
    fftwf_destroy_plan(ifft11);
    fftwf_destroy_plan(ifft12);
    fftwf_destroy_plan(ifft13);
    fftwf_destroy_plan(fft21);
    fftwf_destroy_plan(ifft2);

    color_image_delete(img_pad);

    fftwf_free(ina1);
    fftwf_free(ina2);
    fftwf_free(ina3);
    fftwf_free(inb1);
    fftwf_free(inb2);
    fftwf_free(inb3);
    fftwf_free(out1);
    fftwf_free(out2);
    fftwf_free(out3);
    fftwf_free(fx);
    fftwf_free(fy);
    fftwf_free(gfc);

    fftw_unlock();
}

/*--------------------------------------------------------------------------*/

static void down_N(float *res, image_t *src, int N)
{
    int i, j, k, l;

    int *nx = (int *) malloc((N+1)*sizeof(int));
    int *ny = (int *) malloc((N+1)*sizeof(int));

    for(i = 0; i < N+1; i++)
    {
        nx[i] = i*src->width/(N);
        ny[i] = i*src->height/(N);
    }

    for(l = 0; l < N; l++)
    {
        for(k = 0; k < N; k++)
        {
            float mean = 0.0f;
			float denom = 0.0f;

            for(j = ny[l]; j < ny[l+1]; j++)
            {
                for(i = nx[k]; i < nx[k+1]; i++) {
                    mean += src->data[j*src->stride+i];
                }
            }

            denom = (float)(ny[l+1]-ny[l])*(nx[k+1]-nx[k]);

            res[k*N+l] = mean / denom;
        }
    }

    free(nx);
    free(ny);
}

/*--------------------------------------------------------------------------*/

static void color_down_N(float *res, color_image_t *src, int N, int c)
{
    int i, j, k, l;

    int *nx = (int *) malloc((N+1)*sizeof(int));
    int *ny = (int *) malloc((N+1)*sizeof(int));

    for(i = 0; i < N+1; i++)
    {
        nx[i] = i*src->width/(N);
        ny[i] = i*src->height/(N);
    }

    for(l = 0; l < N; l++)
    {
        for(k = 0; k < N; k++)
        {
            float mean = 0.0f;

            float *ptr;
            float denom = 0.0f;
			switch(c)
            {
                case 0:
                    ptr = src->c1;
                    break;

                case 1:
                    ptr = src->c2;
                    break;

                case 2:
                    ptr = src->c3;
                    break;

                default:
                    return;
            }

            for(j = ny[l]; j < ny[l+1]; j++)
            {
                for(i = nx[k]; i < nx[k+1]; i++)
                {
                    mean += ptr[j*src->width+i];
                }
            }

            denom = (float)(ny[l+1]-ny[l])*(nx[k+1]-nx[k]);

            res[k*N+l] = mean / denom;
            assert(finite(res[k*N+l]));
        }
    }

    free(nx);
    free(ny);
}

/*--------------------------------------------------------------------------*/

static float *gist_gabor(image_t *src, const int w, image_list_t *G)
{
    int i, j, k;
	int width = 0;
    int height = 0;
    int stride = 0;

    float *res = NULL;

    fftwf_complex *in1 = NULL;
    fftwf_complex *in2 = NULL;
    fftwf_complex *out1 = NULL;
    fftwf_complex *out2 = NULL;
    
	fftwf_plan fft;
    fftwf_plan ifft;

	fftw_lock();

    
    /* Get sizes */
    width = src->width;
    height = src->height;
    stride = src->stride;

    res = (float *) malloc(w*w*G->size*sizeof(float));

    in1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    in2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    out1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    out2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));

    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            in1[j*width + i][0] = src->data[j*stride+i];
            in1[j*width + i][1] = 0.0f;
        }
    }

    /* FFT */
    fft = fftwf_plan_dft_2d(width, height, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    ifft = fftwf_plan_dft_2d(width, height, out2, in2, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_unlock();
    fftwf_execute(fft);

    for(k = 0; k < G->size; k++)
    {
        for(j = 0; j < height; j++)
        {
            for(i = 0; i < width; i++)
            {
                out2[j*width+i][0] = out1[j*width+i][0] * G->data[k]->data[j*stride+i];
                out2[j*width+i][1] = out1[j*width+i][1] * G->data[k]->data[j*stride+i];
            }
        }

        fftwf_execute(ifft);

        for(j = 0; j < height; j++)
        {
            for(i = 0; i < width; i++) {
                src->data[j*stride+i] = sqrt(in2[j*width+i][0]*in2[j*width+i][0]+in2[j*width+i][1]*in2[j*width+i][1])/(width*height);
            }
        }

        down_N(res+k*w*w, src, w);
    }

    fftw_lock();

    fftwf_destroy_plan(fft);
    fftwf_destroy_plan(ifft);

    fftwf_free(in1);
    fftwf_free(in2);
    fftwf_free(out1);
    fftwf_free(out2);

    fftw_unlock();

    return res;
}

/*--------------------------------------------------------------------------*/

static float *color_gist_gabor(color_image_t *src, const int w, image_list_t *G)
{
    
    int i, j, k;
	int width = src->width;
    int height = src->height;

    float *res = (float *) malloc(3*w*w*G->size*sizeof(float));

    fftwf_complex *ina1  = NULL;
    fftwf_complex *ina2  = NULL;
    fftwf_complex *ina3  = NULL;
    fftwf_complex *inb1  = NULL;
    fftwf_complex *inb2  = NULL;
    fftwf_complex *inb3  = NULL;
    fftwf_complex *outa1 = NULL;
    fftwf_complex *outa2 = NULL;
    fftwf_complex *outa3 = NULL;
    fftwf_complex *outb1 = NULL;
    fftwf_complex *outb2 = NULL;
    fftwf_complex *outb3 = NULL;

	fftwf_plan fft1;
    fftwf_plan fft2;
    fftwf_plan fft3;
    fftwf_plan ifft1;
    fftwf_plan ifft2;
    fftwf_plan ifft3;

	fftw_lock();

    /* Get sizes */
    width = src->width;
    height = src->height;

    res = (float *) malloc(3*w*w*G->size*sizeof(float));

    ina1  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    ina2  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    ina3  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    inb1  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    inb2  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    inb3  = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    outa1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    outa2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    outa3 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    outb1 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    outb2 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    outb3 = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));

    for(j = 0; j < height; j++)
    {
        for(i = 0; i < width; i++)
        {
            ina1[j*width+i][0] = src->c1[j*width+i];
            ina2[j*width+i][0] = src->c2[j*width+i];
            ina3[j*width+i][0] = src->c3[j*width+i];

            ina1[j*width+i][1] = 0.0f;
            ina2[j*width+i][1] = 0.0f;
            ina3[j*width+i][1] = 0.0f;
        }
    }

    /* FFT */
    fft1  = fftwf_plan_dft_2d(width, height, ina1, outa1, FFTW_FORWARD, FFTW_ESTIMATE);
    fft2  = fftwf_plan_dft_2d(width, height, ina2, outa2, FFTW_FORWARD, FFTW_ESTIMATE);
    fft3  = fftwf_plan_dft_2d(width, height, ina3, outa3, FFTW_FORWARD, FFTW_ESTIMATE);
    ifft1 = fftwf_plan_dft_2d(width, height, outb1, inb1, FFTW_BACKWARD, FFTW_ESTIMATE);
    ifft2 = fftwf_plan_dft_2d(width, height, outb2, inb2, FFTW_BACKWARD, FFTW_ESTIMATE);
    ifft3 = fftwf_plan_dft_2d(width, height, outb3, inb3, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_unlock();
    fftwf_execute(fft1);
    fftwf_execute(fft2);
    fftwf_execute(fft3);

    for(k = 0; k < G->size; k++)
    {
        for(j = 0; j < height; j++)
        {
            for(i = 0; i < width; i++)
            {
                outb1[j*width+i][0] = outa1[j*width+i][0] * G->data[k]->data[j*G->data[k]->stride+i];
                outb2[j*width+i][0] = outa2[j*width+i][0] * G->data[k]->data[j*G->data[k]->stride+i];
                outb3[j*width+i][0] = outa3[j*width+i][0] * G->data[k]->data[j*G->data[k]->stride+i];
                outb1[j*width+i][1] = outa1[j*width+i][1] * G->data[k]->data[j*G->data[k]->stride+i];
                outb2[j*width+i][1] = outa2[j*width+i][1] * G->data[k]->data[j*G->data[k]->stride+i];
                outb3[j*width+i][1] = outa3[j*width+i][1] * G->data[k]->data[j*G->data[k]->stride+i];
            }
        }

        fftwf_execute(ifft1);
        fftwf_execute(ifft2);
        fftwf_execute(ifft3);

        for(j = 0; j < height; j++)
        {
            for(i = 0; i < width; i++)
            {
                src->c1[j*width+i] = sqrt(inb1[j*width+i][0]*inb1[j*width+i][0]+inb1[j*width+i][1]*inb1[j*width+i][1])/(width*height);
                src->c2[j*width+i] = sqrt(inb2[j*width+i][0]*inb2[j*width+i][0]+inb2[j*width+i][1]*inb2[j*width+i][1])/(width*height);
                src->c3[j*width+i] = sqrt(inb3[j*width+i][0]*inb3[j*width+i][0]+inb3[j*width+i][1]*inb3[j*width+i][1])/(width*height);
            }
        }

        color_down_N(res+0*G->size*w*w+k*w*w, src, w, 0);
        color_down_N(res+1*G->size*w*w+k*w*w, src, w, 1);
        color_down_N(res+2*G->size*w*w+k*w*w, src, w, 2);
    }

    fftw_lock();

    fftwf_destroy_plan(fft1);
    fftwf_destroy_plan(fft2);
    fftwf_destroy_plan(fft3);
    fftwf_destroy_plan(ifft1);
    fftwf_destroy_plan(ifft2);
    fftwf_destroy_plan(ifft3);

    fftwf_free(ina1);
    fftwf_free(ina2);
    fftwf_free(ina3);
    fftwf_free(inb1);
    fftwf_free(inb2);
    fftwf_free(inb3);
    fftwf_free(outa1);
    fftwf_free(outa2);
    fftwf_free(outa3);
    fftwf_free(outb1);
    fftwf_free(outb2);
    fftwf_free(outb3);

    fftw_unlock();

    return res;
}

/*--------------------------------------------------------------------------*/

float *bw_gist(image_t *src, int w, int a, int b, int c)
{
    int orientationsPerScale[3];

    orientationsPerScale[0] = a;
    orientationsPerScale[1] = b;
    orientationsPerScale[2] = c;

    return bw_gist_scaletab(src,w,3,orientationsPerScale);
}

float *bw_gist_scaletab(image_t *src, int w, int n_scale, const int *n_orientation)
{
    int i;

    if(src->width < 8 || src->height < 8)
    {
        fprintf(stderr, "Error: bw_gist_scaletab() - Image not big enough !\n");
        return NULL;
    }

    int numberBlocks = w;
    int tot_oris=0;
    for(i=0;i<n_scale;i++) tot_oris+=n_orientation[i];

    image_t *img = image_cpy(src);
    image_list_t *G = create_gabor(n_scale, n_orientation, img->width, img->height);

    prefilt(img, 4);

    float *g = gist_gabor(img, numberBlocks, G);

    for(i = 0; i < tot_oris*w*w; i++)
    {
        if(!_finite(g[i]))
        {
            fprintf(stderr, "Error: bw_gist_scaletab() - descriptor not valid (nan or inf)\n");
            free(g); g=NULL;
            break;
        }
    }

    image_list_delete(G);
    image_delete(img);

    return g;
}

/*--------------------------------------------------------------------------*/

float *color_gist(color_image_t *src, int w, int a, int b, int c) {  
    int orientationsPerScale[3];

    orientationsPerScale[0] = a;
    orientationsPerScale[1] = b;
    orientationsPerScale[2] = c;

    return color_gist_scaletab(src,w,3,orientationsPerScale);

}

float *color_gist_scaletab(color_image_t *src, int w, int n_scale, const int *n_orientation) 
{
    int i;

    if(src->width < 8 || src->height < 8)
    {
        fprintf(stderr, "Error: color_gist_scaletab() - Image not big enough !\n");
        return NULL;
    }

    int numberBlocks = w;
    int tot_oris=0;
    for(i=0;i<n_scale;i++) tot_oris+=n_orientation[i];

    color_image_t *img = color_image_cpy(src);

    image_list_t *G = create_gabor(n_scale, n_orientation, img->width, img->height);

    color_prefilt(img, 4);

    float *g = color_gist_gabor(img, numberBlocks, G);  
    
    for(i = 0; i < tot_oris*w*w*3; i++)
    {
        if(!_finite(g[i]))
        {
            fprintf(stderr, "Error: color_gist_scaletab() - descriptor not valid (nan or inf)\n");
            free(g); g=NULL;
            break;
        }
    }

    image_list_delete(G);
    color_image_delete(img);

    return g;
}

#ifndef STANDALONE_GIST

/*--------------------------------------------------------------------------*/

local_desc_list_t *descriptor_bw_gist_cpu(image_t *src, int a, int b, int c, int w)
{
    local_desc_list_t *desc_list = local_desc_list_new();

    float *desc_raw = bw_gist(src, w, a, b, c); 

    local_desc_t *desc = local_desc_new();

    desc->desc_size = (a+b+c)*w*w;
    desc->desc = (float *) malloc(desc->desc_size*sizeof(float));
    memcpy(desc->desc, desc_raw, desc->desc_size*sizeof(float));
    local_desc_list_append(desc_list, desc);

    free(desc_raw);

    return desc_list;
}

/*--------------------------------------------------------------------------*/

local_desc_list_t *descriptor_color_gist_cpu(color_image_t *src, int a, int b, int c, int w)
{
    local_desc_list_t *desc_list = local_desc_list_new();

    float *desc_raw = color_gist(src, w, a, b, c);

    local_desc_t *desc = local_desc_new();

    desc->desc_size = 3*(a+b+c)*w*w;
    desc->desc = (float *) malloc(desc->desc_size*sizeof(float));
    memcpy(desc->desc, desc_raw, desc->desc_size*sizeof(float));
    local_desc_list_append(desc_list, desc);

    free(desc_raw);

    return desc_list;
}

/*--------------------------------------------------------------------------*/

#endif 

#endif

/*--------------------------------------------------------------------------*/


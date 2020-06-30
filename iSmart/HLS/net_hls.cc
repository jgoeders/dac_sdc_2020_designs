#include "net_hls.h"
#include <math.h>
#include <fstream>
#include <hls_math.h>
#include <ap_fixed.h>
#include <string.h>

using namespace std;

#define DDR_OFFSET 0 
#define DW1_POOL_OFFSET 524288
#define DW2_POOL_OFFSET 786432


// feature map buffers
FIX_FM FM_buf1[32][44][84];
FIX_FM FM_buf2[32][44][84];
FIX_FM FM_buf3[32][44][84];
FIX_FM FM_buf4[32][44][84];
FIX_FM_acc FM_buf_acc[32][44][84];

// weight buffers
FIX_WT weight_buf_1x1[4][32][32];
FIX_WT weight_buf_3x3[4][32][3][3];
FIX_WT bias_buf[4][32];



void compute_bounding_box(float predict_box[4][5], int constant[4][3])
{
    FIX_32_4 conf_thresh = -100.0;
    int conf_j = 0;
    int conf_m = 0;
    int conf_n = 0;
    FIX_32_4 conf_box1 = 0.0;
    FIX_32_4 conf_box2 = 0.0;
    //int h = 20;
    //int w = 40;

    //// 0
    for(int m = 1; m <= 20; m++){
        for(int n = 1; n <= 40; n++){
            //conf_box1 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf_acc[4][m][n]));
            conf_box1 = FM_buf_acc[4][m][n];
            if(conf_box1 > conf_thresh){
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

            }
        }
    }

    for(int m = 1; m <= 20; m++){
        for(int n = 1; n <= 40; n++){
            //conf_box2 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf_acc[9][m][n]));
        	conf_box2 = FM_buf_acc[9][m][n];
            if(conf_box2 > conf_thresh){
                conf_thresh = conf_box2;
                conf_j = 1;
                conf_m = m;
                conf_n = n;
            }
        }
    }

	if( conf_j == 0 ) {
		// first bounding box
		predict_box[0][0] = FM_buf_acc[0][conf_m][conf_n];
		predict_box[0][1] = FM_buf_acc[1][conf_m][conf_n];
		predict_box[0][2] = FM_buf_acc[2][conf_m][conf_n];
		predict_box[0][3] = FM_buf_acc[3][conf_m][conf_n];
		predict_box[0][4] = conf_thresh;

	}
	else if( conf_j == 1 ) {
		// second bounding box
		predict_box[0][0] = FM_buf_acc[5][conf_m][conf_n];
		predict_box[0][1] = FM_buf_acc[6][conf_m][conf_n];
		predict_box[0][2] = FM_buf_acc[7][conf_m][conf_n];
		predict_box[0][3] = FM_buf_acc[8][conf_m][conf_n];
		predict_box[0][4] = conf_thresh;
	}

    constant[0][0] = conf_j;
    constant[0][1] = conf_n-1;
    constant[0][2] = conf_m-1;

    //printf("0:  conf_m: %d, conf_n:%d\n", conf_m-1, conf_n-1);



    //// 1
    conf_thresh = -100.0;
    conf_j = 0;
    conf_m = 0;
    conf_n = 0;
    conf_box1 = 0.0;
    conf_box2 = 0.0;

    for(int m = 1; m <= 20; m++){
        for(int n = 43; n <= 82; n++){
            //conf_box1 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf_acc[4][m][n]));
            conf_box1 = FM_buf_acc[4][m][n];
            if(conf_box1 > conf_thresh){
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

            }
        }
    }

    for(int m = 1; m <= 20; m++){
        for(int n = 43; n <= 82; n++){
            //conf_box2 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf_acc[9][m][n]));
        	conf_box2 = FM_buf_acc[9][m][n];
            if(conf_box2 > conf_thresh){
                conf_thresh = conf_box2;
                conf_j = 1;
                conf_m = m;
                conf_n = n;
            }
        }
    }

	if( conf_j == 0 ) {
		// first bounding box
		predict_box[1][0] = FM_buf_acc[0][conf_m][conf_n];
		predict_box[1][1] = FM_buf_acc[1][conf_m][conf_n];
		predict_box[1][2] = FM_buf_acc[2][conf_m][conf_n];
		predict_box[1][3] = FM_buf_acc[3][conf_m][conf_n];
		predict_box[1][4] = conf_thresh;

	}
	else if( conf_j == 1 ) {
		// second bounding box
		predict_box[1][0] = FM_buf_acc[5][conf_m][conf_n];
		predict_box[1][1] = FM_buf_acc[6][conf_m][conf_n];
		predict_box[1][2] = FM_buf_acc[7][conf_m][conf_n];
		predict_box[1][3] = FM_buf_acc[8][conf_m][conf_n];
		predict_box[1][4] = conf_thresh;
	}

    constant[1][0] = conf_j;
    constant[1][1] = conf_n-43;
    constant[1][2] = conf_m-1;

    //printf("1:  conf_m: %d, conf_n:%d\n", conf_m-1, conf_n-1);


    //// 2
    conf_thresh = -100.0;
    conf_j = 0;
    conf_m = 0;
    conf_n = 0;
    conf_box1 = 0.0;
    conf_box2 = 0.0;

    for(int m = 23; m <= 42; m++){
        for(int n = 1; n <= 40; n++){
            //conf_box1 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf_acc[4][m][n]));
            conf_box1 = FM_buf_acc[4][m][n];
            if(conf_box1 > conf_thresh){
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

            }
        }
    }

    for(int m = 23; m <= 42; m++){
        for(int n = 1; n <= 40; n++){
            //conf_box2 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf_acc[9][m][n]));
        	conf_box2 = FM_buf_acc[9][m][n];
            if(conf_box2 > conf_thresh){
                conf_thresh = conf_box2;
                conf_j = 1;
                conf_m = m;
                conf_n = n;
            }
        }
    }

	if( conf_j == 0 ) {
		// first bounding box
		predict_box[2][0] = FM_buf_acc[0][conf_m][conf_n];
		predict_box[2][1] = FM_buf_acc[1][conf_m][conf_n];
		predict_box[2][2] = FM_buf_acc[2][conf_m][conf_n];
		predict_box[2][3] = FM_buf_acc[3][conf_m][conf_n];
		predict_box[2][4] = conf_thresh;

	}
	else if( conf_j == 1 ) {
		// second bounding box
		predict_box[2][0] = FM_buf_acc[5][conf_m][conf_n];
		predict_box[2][1] = FM_buf_acc[6][conf_m][conf_n];
		predict_box[2][2] = FM_buf_acc[7][conf_m][conf_n];
		predict_box[2][3] = FM_buf_acc[8][conf_m][conf_n];
		predict_box[2][4] = conf_thresh;
	}

    constant[2][0] = conf_j;
    constant[2][1] = conf_n-1;
    constant[2][2] = conf_m-23;

    //printf("2:  conf_m: %d, conf_n:%d\n", conf_m-1, conf_n-1);



    //// 3
    conf_thresh = -100.0;
    conf_j = 0;
    conf_m = 0;
    conf_n = 0;
    conf_box1 = 0.0;
    conf_box2 = 0.0;

    for(int m = 23; m <= 42; m++){
        for(int n = 43; n <= 82; n++){
            //conf_box1 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf_acc[4][m][n]));
            conf_box1 = FM_buf_acc[4][m][n];
            if(conf_box1 > conf_thresh){
				conf_thresh = conf_box1;
				conf_j = 0;
				conf_m = m;
				conf_n = n;

            }
        }
    }

    for(int m = 23; m <= 42; m++){
        for(int n = 43; n <= 82; n++){
            //conf_box2 = (FIX_32_25)1 / ((FIX_32_25)1 + my_exp_fix(-FM_buf_acc[9][m][n]));
        	conf_box2 = FM_buf_acc[9][m][n];
            if(conf_box2 > conf_thresh){
                conf_thresh = conf_box2;
                conf_j = 1;
                conf_m = m;
                conf_n = n;
            }
        }
    }

	if( conf_j == 0 ) {
		// first bounding box
		predict_box[3][0] = FM_buf_acc[0][conf_m][conf_n];
		predict_box[3][1] = FM_buf_acc[1][conf_m][conf_n];
		predict_box[3][2] = FM_buf_acc[2][conf_m][conf_n];
		predict_box[3][3] = FM_buf_acc[3][conf_m][conf_n];
		predict_box[3][4] = conf_thresh;

	}
	else if( conf_j == 1 ) {
		// second bounding box
		predict_box[3][0] = FM_buf_acc[5][conf_m][conf_n];
		predict_box[3][1] = FM_buf_acc[6][conf_m][conf_n];
		predict_box[3][2] = FM_buf_acc[7][conf_m][conf_n];
		predict_box[3][3] = FM_buf_acc[8][conf_m][conf_n];
		predict_box[3][4] = conf_thresh;
	}

    constant[3][0] = conf_j;
    constant[3][1] = conf_n-43;
    constant[3][2] = conf_m-23;

    //printf("3:  conf_m: %d, conf_n:%d\n", conf_m-1, conf_n-1);

}


inline FIX_FM relu_single( FIX_FM d ) {
	if( d > 6 )
		return 6;
	if( d < 0 )
		return 0;
	return d;
}


inline FIX_FM_acc fmrg_acc(uint256 DATA, int offset){
	FIX_FM_acc d = 0;
	d.range(FM_RG,0) = DATA.range(offset + FM_RG, offset);
	return d;
}

inline FIX_FM fmrg(uint256 DATA, int offset){
	FIX_FM d = 0;
	d.range(FM_RG,0) = DATA.range(offset + FM_RG, offset);
	return d;
}




void relu_copy_buf_to_DDR( uint256* dest, int buf_id, FIX_FM src[32][44][84], int offset_h, int offset_w)
{
	uint256* dest_ptr = dest + 44*84*buf_id + (offset_h*22)*84 + (offset_w*42);

	for(int h = 0; h < 44; h++) {


		for(int w = 0; w < 84; w++) {
#pragma HLS pipeline

			uint256 DATA = 0;
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				FIX_FM d = relu_single(src[c][h][w]);
				DATA.range(FM_RG + 8*c, 8*c) = d.range(FM_RG, 0);
			}
			dest_ptr[w].range(255, 0) = DATA.range(255, 0);
		}

		dest_ptr += 84;
	}
}




void relu_copy_buf_to_DDR_acc( uint256* dest, int buf_id, FIX_FM_acc src[32][44][84], int offset_h, int offset_w)
{
	uint256* dest_ptr = dest + 44*84*buf_id + (offset_h*22)*84 + (offset_w*42);

	for(int h = 0; h < 44; h++) {

		for(int w = 0; w < 84; w++) {
#pragma HLS pipeline

			uint256 DATA = 0;
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				FIX_FM d = relu_single((FIX_FM)src[c][h][w]);
				DATA.range(FM_RG + 8*c, 8*c) = d.range(FM_RG, 0);
			}
			dest_ptr[w].range(255, 0) = DATA.range(255, 0);
		}

		dest_ptr += 84;
	}
}



void load_buf_from_DDR( FIX_FM dest[32][44][84], uint256* src, int buf_id)
{
	uint256* src_ptr = src + 44*84*buf_id;

	for(int h = 0; h < 44; h++) {

		for(int w = 0; w < 84; w++) {
#pragma HLS pipeline II=1

			uint256 DATA= src_ptr[w];
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				dest[c][h][w] = fmrg(DATA, c*8);
			}
		}
		src_ptr += 84;
	}
}




void load_weight_1x1_from_axi( FIX_WT dest[32][32], uint512 src[32])
{

	for(int ci = 0; ci < 32; ci++) {
#pragma HLS pipeline
		uint512 DATA = 0;
		DATA.range(511, 0) = src[ci].range(511, 0);
		for(int co = 0; co < 32; co++) {
#pragma HLS unroll
			dest[co][ci].range(WT_RG, 0) = DATA.range(WT_RG + co*16, co*16);
		}
	}
}



void load_weight_3x3_from_axi( FIX_WT dest[32][3][3], uint512 src[3][3])
{
	for(int m = 0; m < 3; m++) {
		for(int n = 0; n < 3; n++) {
#pragma HLS pipeline
			uint512 DATA = 0;
			DATA.range(511, 0) = src[m][n].range(511, 0);
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll

				dest[c][m][n].range(WT_RG, 0) = DATA.range(WT_RG + c*16, c*16);
			}
		}
	}
}



void load_bias_from_axi(FIX_WT dest[32], uint512 src)
{
	for(int c = 0; c < 32; c++) {
#pragma HLS unroll
		dest[c].range(WT_RG, 0) = src.range(WT_RG + c*16, c*16);
	}
}


void set_bias_1x1( FIX_FM_acc buf[32][44][84], FIX_WT bias[32])
{
#pragma HLS array_partition variable=bias dim=1 complete

	for(int h = 1; h <= 42; h+=2) {
		for(int w = 1; w <= 82; w++) {
#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				buf[c][h  ][w] = bias[c];
				buf[c][h+1][w] = bias[c];
			}
		}
	}
}


void set_bias_3x3( FIX_FM buf[32][44][84], FIX_WT bias[32])
{
#pragma HLS array_partition variable=bias dim=1 complete

	for(int h = 1; h <= 42; h+=2) {
		for(int w = 1; w <= 82; w++) {
#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				buf[c][h  ][w] = bias[c];
				buf[c][h+1][w] = bias[c];
			}
		}
	}
}

void Relu_Convert_FIX_set_bias_3x3( FIX_FM_acc buf_acc[32][44][84], FIX_FM buf_bottom[32][44][84],
							   FIX_FM buf_top[32][44][84], FIX_WT bias[32])
{
#pragma HLS array_partition variable=bias dim=1 complete
#pragma HLS array_partition variable=buf_acc dim=1 complete
#pragma HLS array_partition variable=buf_bottom dim=1 complete
#pragma HLS array_partition variable=buf_top dim=1 complete

	for(int h = 1; h <= 42; h++) {
		for(int w = 1; w <= 82; w++) {
#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				buf_bottom[c][h][w] = relu_single((FIX_FM)buf_acc[c][h][w]);
				buf_top[c][h][w] = bias[c];
			}
		}
	}
}


void Relu_Convert_FIX( FIX_FM_acc buf_acc[32][44][84], FIX_FM buf_bottom[32][44][84])
{
// #pragma HLS array_partition variable=bias dim=1 complete
#pragma HLS array_partition variable=buf_acc dim=1 complete
#pragma HLS array_partition variable=buf_bottom dim=1 complete
// #pragma HLS array_partition variable=buf_top dim=1 complete

	for(int h = 1; h <= 42; h++) {
		for(int w = 1; w <= 82; w++) {
#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				buf_bottom[c][h][w] = relu_single((FIX_FM)buf_acc[c][h][w]);
				// buf_top[c][h][w] = bias[c];
			}
		}
	}
}



void load_dw1_pool_from_DDR( uint256* ddr_dw1_pool_burst,
							 FIX_FM buf[32][44][84],
							 int ch, int col, int row, int offset_h, int offset_w)
{
	uint256* ddr_dw1_pool_burst_ptr =ddr_dw1_pool_burst + ch*82*2*162*2 + (col*40 + offset_h*2)*162*2 + (row*80 + offset_w*2);

	for(int h = 0; h < 42; h++) {
		for(int w = 0; w < 82; w++) {
#pragma HLS pipeline
			uint256 DATA = 0;
			DATA.range(255, 0) = ddr_dw1_pool_burst_ptr[w].range(255, 0);
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				buf[c][h][w] = fmrg(DATA, c*8);
			}
		}
		ddr_dw1_pool_burst_ptr += 162*2;
	}
}



void load_dw2_pool_from_DDR( uint256* ddr_dw2_pool_burst,
							 FIX_FM buf[32][44][84],
							 int ch, int col, int row, int offset_h, int offset_w)
{
	uint256* ddr_dw2_pool_burst_ptr = ddr_dw2_pool_burst + ch*42*2*82*2 + (col*40 + offset_h*2)*82*2 + (row*80 + offset_w*2);

	for(int h = 0; h < 42; h++) {
		for(int w = 0; w < 82; w++) {
#pragma HLS pipeline
			uint256 DATA = 0;
			DATA.range(255, 0) = ddr_dw2_pool_burst_ptr[w].range(255, 0);
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				buf[c][h][w] = fmrg(DATA, c*8);
			}
		}
		ddr_dw2_pool_burst_ptr += 82*2;
	}
}




void clear_buffer( FIX_FM_acc buf[32][44][84] )
{
	for(int h = 0; h < 44; h+=2) {
		for(int w = 0; w < 84; w++) {
#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				buf[c][h  ][w] = 0;
				buf[c][h+1][w] = 0;
			}
		}
	}
}


FIX_FM img_norm_ch[256] = {
		-2.000000, -1.984314, -1.968627, -1.952941, -1.937255, -1.921569, -1.905882, -1.890196, -1.874510, -1.858824, -1.843137, -1.827451, -1.811765, -1.796078, -1.780392, -1.764706, -1.749020,
		-1.733333, -1.717647, -1.701961, -1.686275, -1.670588, -1.654902, -1.639216, -1.623529, -1.607843, -1.592157, -1.576471, -1.560784, -1.545098, -1.529412, -1.513725, -1.498039,
		-1.482353, -1.466667, -1.450980, -1.435294, -1.419608, -1.403922, -1.388235, -1.372549, -1.356863, -1.341176, -1.325490, -1.309804, -1.294118, -1.278431, -1.262745, -1.247059,
		-1.231373, -1.215686, -1.200000, -1.184314, -1.168627, -1.152941, -1.137255, -1.121569, -1.105882, -1.090196, -1.074510, -1.058824, -1.043137, -1.027451, -1.011765, -0.996078,
		-0.980392, -0.964706, -0.949020, -0.933333, -0.917647, -0.901961, -0.886275, -0.870588, -0.854902, -0.839216, -0.823529, -0.807843, -0.792157, -0.776471, -0.760784, -0.745098,
		-0.729412, -0.713725, -0.698039, -0.682353, -0.666667, -0.650980, -0.635294, -0.619608, -0.603922, -0.588235, -0.572549, -0.556863, -0.541176, -0.525490, -0.509804, -0.494118,
		-0.478431, -0.462745, -0.447059, -0.431373, -0.415686, -0.400000, -0.384314, -0.368627, -0.352941, -0.337255, -0.321569, -0.305882, -0.290196, -0.274510, -0.258824, -0.243137,
		-0.227451, -0.211765, -0.196078, -0.180392, -0.164706, -0.149020, -0.133333, -0.117647, -0.101961, -0.086275, -0.070588, -0.054902, -0.039216, -0.023529, -0.007843, 0.007843,
		0.023529, 0.039216, 0.054902, 0.070588, 0.086275, 0.101961, 0.117647, 0.133333, 0.149020, 0.164706, 0.180392, 0.196078, 0.211765, 0.227451, 0.243137, 0.258824,
		0.274510, 0.290196, 0.305882, 0.321569, 0.337255, 0.352941, 0.368627, 0.384314, 0.400000, 0.415686, 0.431373, 0.447059, 0.462745, 0.478431, 0.494118, 0.509804,
		0.525490, 0.541176, 0.556863, 0.572549, 0.588235, 0.603922, 0.619608, 0.635294, 0.650980, 0.666667, 0.682353, 0.698039, 0.713725, 0.729412, 0.745098, 0.760784,
		0.776471, 0.792157, 0.807843, 0.823529, 0.839216, 0.854902, 0.870588, 0.886275, 0.901961, 0.917647, 0.933333, 0.949020, 0.964706, 0.980392, 0.996078, 1.011765,
		1.027451, 1.043137, 1.058824, 1.074510, 1.090196, 1.105882, 1.121569, 1.137255, 1.152941, 1.168627, 1.184314, 1.200000, 1.215686, 1.231373, 1.247059, 1.262745,
		1.278431, 1.294118, 1.309804, 1.325490, 1.341176, 1.356863, 1.372549, 1.388235, 1.403922, 1.419608, 1.435294, 1.450980, 1.466667, 1.482353, 1.498039, 1.513725,
		1.529412, 1.545098, 1.560784, 1.576471, 1.592157, 1.607843, 1.623529, 1.639216, 1.654902, 1.670588, 1.686275, 1.701961, 1.717647, 1.733333, 1.749020, 1.764706,
		1.780392, 1.796078, 1.811765, 1.827451, 1.843137, 1.858824, 1.874510, 1.890196, 1.905882, 1.921569, 1.937255, 1.952941, 1.968627, 1.984314, 2.000000
};



void load_image_chunk_norm(FIX_FM img_buf[32][44][84], int image_in_raw_pad_burst[3*162*2*324*2*2],
							int col, int row, int offset_h = 0, int offset_w = 0)
{
	int* image_in_raw_pad_burst_ptr;

	image_in_raw_pad_burst_ptr = image_in_raw_pad_burst + (col*40 + offset_h*2)*322*2 + row*80 + offset_w*2;
	for(int i = 0; i < 44; i++) {
		for(int j = 0; j < 84; j++) {
#pragma HLS pipeline
				img_buf[0][i][j] = img_norm_ch[((image_in_raw_pad_burst_ptr[j]))];
		}
		image_in_raw_pad_burst_ptr += 322*2;
	}

	image_in_raw_pad_burst_ptr = image_in_raw_pad_burst + 1*162*2*322*2 + (col*40 + offset_h*2)*322*2 + row*80 + offset_w*2;
	for(int i = 0; i < 44; i++) {
		for(int j = 0; j < 84; j++) {
#pragma HLS pipeline
				img_buf[1][i][j] = img_norm_ch[((image_in_raw_pad_burst_ptr[j]))];
		}
		image_in_raw_pad_burst_ptr += 322*2;
	}

	image_in_raw_pad_burst_ptr = image_in_raw_pad_burst + 2*162*2*322*2 + (col*40 + offset_h*2)*322*2 + row*80 + offset_w*2;
	for(int i = 0; i < 44; i++) {
		for(int j = 0; j < 84; j++) {
#pragma HLS pipeline
				img_buf[2][i][j] = img_norm_ch[((image_in_raw_pad_burst_ptr[j]))];
		}
		image_in_raw_pad_burst_ptr += 322*2;
	}
}



void Relu6_3x3( FIX_FM buf[32][44][84])
{
	for(int h = 1; h <= 42; h+=2) {
		for(int w = 1; w <= 82; w++) {
#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
#pragma HLS unroll
				buf[c][h  ][w] = relu_single(buf[c][h  ][w]);
				buf[c][h+1][w] = relu_single(buf[c][h+1][w]);
			}
		}
	}
}





inline FIX_FM relu_max(FIX_FM a, FIX_FM b, FIX_FM c, FIX_FM d)
{
	FIX_FM t1, t2;

	if(a > b) t1 = relu_single(a);
	else t1 = relu_single(b);

	if(c > d) t2 = relu_single(c);
	else t2 = relu_single(d);

	if(t1 > t2) return t1;
	else return t2;
}




void Relu_Max_Pooling(
		 FIX_FM_acc buf_in[32][44][84],
		 uint256* ddr_buf_merge, 
         int buf_id,
		 int ch, int col, int row, int offset_h, int offset_w, int layer)
{
#pragma HLS array_partition variable=buf_in dim=1 complete

	int ddr_offfset=0;
	int ddr_step= 0;


	if(layer==1)
	{
		ddr_offfset= DW1_POOL_OFFSET+ch*82*2*162*2 + (1 + col*20 + offset_h*2)*162*2 + (row*40 + offset_w*2);
		ddr_step=162*2;
	}
	else if (layer==2)
	{
		ddr_offfset= DW2_POOL_OFFSET+ch*42*2*82*2 + (1 + col*20 + offset_h*2)*82*2 + (row*40 + offset_w*2);
		ddr_step=82*2;
	}
	else
	{
		ddr_offfset=DDR_OFFSET+buf_id*44*84 + (1 + offset_h*22)*84 + offset_w*42;
		ddr_step=84;
	}
	
	uint256* ddr_ptr=ddr_buf_merge+ddr_offfset;
	
    for(int h = 1; h <= 20; h++) {
        for(int w = 1; w <= 40; w++) {
#pragma HLS pipeline II=1
            uint256 DATA = 0;
            for(int c = 0; c < 32; c++) {
#pragma HLS unroll
                FIX_FM d = relu_max(buf_in[c][h*2-1][w*2-1], buf_in[c][h*2-1][w*2],
                            buf_in[c][h*2][w*2-1], buf_in[c][h*2][w*2]);
                DATA.range(FM_RG + c*8, c*8) = d.range(FM_RG, 0);
            }
			ddr_ptr[w].range(255, 0) = DATA.range(255, 0);
        }
        ddr_ptr += ddr_step;
    }
}


void load_and_reorg_part( uint256* buf_in, int buf_id,
						  FIX_FM buf_out_1[32][44][84],
						  FIX_FM buf_out_2[32][44][84],
						  FIX_FM buf_out_3[32][44][84],
						  FIX_FM_acc buf_out_4[32][44][84],	// borrow one
						  int offset_h, int offset_w)
{
	#pragma HLS array_partition variable=buf_out_1 dim=1 complete
	#pragma HLS array_partition variable=buf_out_2 dim=1 complete
	#pragma HLS array_partition variable=buf_out_3 dim=1 complete
	#pragma HLS array_partition variable=buf_out_4 dim=1 complete

	uint256* buf_in_ptr = buf_in + buf_id*44*84+84;
	ap_uint<8> h_address_base=offset_h*22;
	ap_uint<8> w_address_base=offset_w*42;
	ap_uint<1> hpingpong=0;
	ap_uint<1> wpingpong=0;
	for(ap_uint<8> h = 1; h <= 40; h++) {
		for(ap_uint<8> w = 1; w <= 80; w++) {
		#pragma HLS pipeline
			uint256 DATA;
			DATA= buf_in_ptr[w];
			ap_uint<8> data_array[32];
			#pragma array_partition variable=data_array complete
			for(int i=0;i<32;i++)
			{
				#pragma HLS unroll
				data_array[i].range(7,0)=DATA.range(i*8+7,i*8);
			}
			ap_uint<8> h_address=h_address_base+(h+1)/2;
			ap_uint<8> w_address=w_address_base+(w+1)/2;
			ap_uint<2> bank_idx_offset= (hpingpong,wpingpong);
			if( bank_idx_offset==0)
			{
				for(int c = 0; c < 8; c++) {
					#pragma HLS unroll
					buf_out_1[c*4+0][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[c]);
					buf_out_2[c*4+0][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[8+c]);
					buf_out_3[c*4+0][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[16+c]);
					buf_out_4[c*4+0][h_address][w_address].range(12,0) = ((ap_uint<5>) 0,data_array[24+c]);
				}
			}
			else if(bank_idx_offset==1 )
			{
				for(int c = 0; c < 8; c++) {
					#pragma HLS unroll
					buf_out_1[c*4+1][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[c]);
					buf_out_2[c*4+1][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[8+c]);
					buf_out_3[c*4+1][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[16+c]);
					buf_out_4[c*4+1][h_address][w_address].range(12,0) = ((ap_uint<5>) 0,data_array[24+c]);
				}
			}
			else if(bank_idx_offset==2 )
			{
				for(int c = 0; c < 8; c++) {
					#pragma HLS unroll
					buf_out_1[c*4+2][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[c]);
					buf_out_2[c*4+2][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[8+c]);
					buf_out_3[c*4+2][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[16+c]);
					buf_out_4[c*4+2][h_address][w_address].range(12,0) = ((ap_uint<5>) 0,data_array[24+c]);
				}
			}
			else if(bank_idx_offset==3 )
			{
				for(int c = 0; c < 8; c++) {
					#pragma HLS unroll
					buf_out_1[c*4+3][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[c]);
					buf_out_2[c*4+3][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[8+c]);
					buf_out_3[c*4+3][h_address][w_address].range(8,0) = ((ap_uint<1>) 0,data_array[16+c]);
					buf_out_4[c*4+3][h_address][w_address].range(12,0) = ((ap_uint<5>) 0,data_array[24+c]);
				}
			}
			wpingpong=1-wpingpong;
		}
		hpingpong=1-hpingpong;
		buf_in_ptr+=84;
	}
}




void load_and_reorg( uint256* buf_1, int buf_id_1, uint256* buf_2, int buf_id_2,
					 uint256* buf_3, int buf_id_3, uint256* buf_4, int buf_id_4,
					 FIX_FM buf_out_1[32][44][84],
					 FIX_FM buf_out_2[32][44][84],
					 FIX_FM buf_out_3[32][44][84],
					 FIX_FM_acc buf_out_4[32][44][84])
{
	load_and_reorg_part( buf_1, buf_id_1, buf_out_1, buf_out_2, buf_out_3, buf_out_4, 0, 0);
	load_and_reorg_part( buf_2, buf_id_2, buf_out_1, buf_out_2, buf_out_3, buf_out_4, 0, 1);
	load_and_reorg_part( buf_3, buf_id_3, buf_out_1, buf_out_2, buf_out_3, buf_out_4, 1, 0);
	load_and_reorg_part( buf_4, buf_id_4, buf_out_1, buf_out_2, buf_out_3, buf_out_4, 1, 1);
}


void local_buf_copy( FIX_FM dest[32][44][84], FIX_FM_acc src[32][44][84])
{
	for(int h = 0; h < 44; h+=2) {
		for(int w = 0; w < 84; w++) {
#pragma HLS pipeline
			for(int c = 0; c < 32; c++) {
				dest[c][h  ][w].range(FM_RG, 0) = src[c][h  ][w].range(FM_RG, 0);
				dest[c][h+1][w].range(FM_RG, 0) = src[c][h+1][w].range(FM_RG, 0);
			}
		}
	}
}


void buffer_copy_debug( float dest[32][44][84], FIX_FM src[32][44][84])
{
	for(int h = 0; h < 44; h++) {
		for(int w = 0; w < 84; w++) {
			for(int c = 0; c < 32; c++) {
				dest[c][h][w] = src[c][h][w].to_float();
			}
		}
	}
}


void buffer_copy_debug_acc( float dest[32][44][84], FIX_FM_acc src[32][44][84])
{
	for(int h = 0; h < 44; h++) {
		for(int w = 0; w < 84; w++) {
			for(int c = 0; c < 32; c++) {
				dest[c][h][w] = src[c][h][w].to_float();
			}
		}
	}
}



void SkyNet(	int image_in_raw_pad[3*162*2*324*2*2],

				uint512 conv_weight_1x1_all[1000][32],
				uint512 conv_weight_3x3_all[1000][3][3],
				uint512 bias_all[500],

				uint256* DDR_buff_merge,
				float predict_boxes[4][5],
				int constant[4][3]
)
{

#pragma HLS INTERFACE m_axi depth=3*162*324*2*2	port=image_in_raw_pad			offset=slave	bundle=IMG

#pragma HLS INTERFACE m_axi depth=306*32*32		port=conv_weight_1x1_all		offset=slave	bundle=BUS512
#pragma HLS INTERFACE m_axi depth=24*32*3*3		port=conv_weight_3x3_all		offset=slave	bundle=BUS512
#pragma HLS INTERFACE m_axi depth=63*32			port=bias_all					offset=slave	bundle=BUS512

#pragma HLS INTERFACE m_axi depth=524288*2		port=DDR_buf_burst				offset=slave	bundle=DDR256

#pragma HLS INTERFACE m_axi depth=5				port=predict_boxes				offset=slave	bundle=BUS32
#pragma HLS INTERFACE m_axi depth=5				port=constant					offset=slave	bundle=BUS32

#pragma HLS INTERFACE s_axilite register	port=return


#pragma HLS ALLOCATION instances=CONV_1x1_bias		 		limit=1 function
#pragma HLS ALLOCATION instances=DW_CONV_3x3_bias	    		limit=1 function
#pragma HLS ALLOCATION instances=Relu_Max_Pooling	    	limit=1 function
#pragma HLS ALLOCATION instances=load_image_chunk_norm		limit=1 function


	int CI_N, CO_N;
	int weight_3x3_index, weight_1x1_index, bias_3x3_index, bias_1x1_index;

	/////////////////////////////// DW1_CONV_3x3 + DW1_CONV_1x1 + POOL ////////////////////////////
	/// DW1_CONV_3x3
	weight_3x3_index = 0;
	bias_3x3_index = 0;
	weight_1x1_index = 0;
	bias_1x1_index = 1;

	CI_N = 32 / 32;
	CO_N = 64 / 32;

	load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index]);
	load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index]);
	load_bias_from_axi(bias_buf[1], bias_all[bias_1x1_index + 0]);
	load_bias_from_axi(bias_buf[2], bias_all[bias_1x1_index + 1]);
	load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + 0]);
	load_weight_1x1_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1_index + 1]);

	for(int row = 0; row < 8; row++) {

		load_image_chunk_norm(FM_buf1, image_in_raw_pad, 0, row, 0/4, row/4);
		for(int col = 0; col < 8; col++) {

			if( col % 2 == 0 ) {
				DW_CONV_3x3_bias(FM_buf1, FM_buf2, weight_buf_3x3[0], bias_buf[0], 1);
				load_image_chunk_norm(FM_buf3, image_in_raw_pad, col+1, row, (col+1)/4, row/4);
			}
			else {
				DW_CONV_3x3_bias(FM_buf3, FM_buf2, weight_buf_3x3[0], bias_buf[0], 1);
				load_image_chunk_norm(FM_buf1, image_in_raw_pad, col+1, row, (col+1)/4, row/4);
			}

			for(int co = 0; co < CO_N; co++) {
				CONV_1x1_bias(FM_buf2, FM_buf_acc, weight_buf_1x1[co],bias_buf[1 + co],1);
				Relu_Max_Pooling(FM_buf_acc, DDR_buff_merge, 0, co, col, row, col/4, row/4, 1);
			}
		}
	}


	printf("DW1 Done\n");


	/////////////////////////////// DW2_CONV_3x3 + DW2_CONV_1x1 + POOL ////////////////////////////
	/// DW2_CONV_3x3
	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;

	load_weight_3x3_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[weight_3x3_index + 0]);
	load_weight_3x3_from_axi(weight_buf_3x3[2], conv_weight_3x3_all[weight_3x3_index + 1]);
	load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + 0]);
	load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + 1]);

	CI_N = 64 / 32;
	CO_N = 96 / 32;

	for(int row = 0; row < 4; row++) {
		for(int col = 0; col < 4; col++) {

			load_dw1_pool_from_DDR(DDR_buff_merge+DW1_POOL_OFFSET, FM_buf1, 0, col, row, col/2, row/2);
			DW_CONV_3x3_bias(FM_buf1, FM_buf2, weight_buf_3x3[1 + 0], bias_buf[0 + 0], 1);

			load_dw1_pool_from_DDR(DDR_buff_merge+DW1_POOL_OFFSET, FM_buf4, 1, col, row, col/2, row/2);
			DW_CONV_3x3_bias(FM_buf4, FM_buf3, weight_buf_3x3[1 + 1], bias_buf[0 + 1], 1);

			for(int co = 0; co < CO_N; co++) {
				load_bias_from_axi(bias_buf[2], bias_all[bias_1x1_index + co]);

				load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + 0 + co * CI_N]);
				CONV_1x1_bias(FM_buf2, FM_buf_acc, weight_buf_1x1[0],bias_buf[2],0,true);

				load_weight_1x1_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1_index + 1 + co * CI_N]);
				CONV_1x1_bias(FM_buf3, FM_buf_acc, weight_buf_1x1[1], bias_buf[2],1,false);

				Relu_Max_Pooling(FM_buf_acc, DDR_buff_merge, 0, co, col, row, col/2, row/2, 2);
			}
		}
	}


	printf("DW2 Done\n");



	/////////////////////////////// DW3_CONV_3x3 + DW3_CONV_1x1 + POOL ////////////////////////////


	/// DW3_CONV_3x3
	// output in DDR_buf[0] - DDR_buf[2]

	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;

	load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + 0]);
	load_weight_3x3_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[weight_3x3_index + 1]);
	load_weight_3x3_from_axi(weight_buf_3x3[2], conv_weight_3x3_all[weight_3x3_index + 2]);
	load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + 0]);
	load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + 1]);
	load_bias_from_axi(bias_buf[2], bias_all[bias_3x3_index + 2]);

	CI_N = 96 / 32;
	CO_N = 192 / 32;
	for(int row = 0; row < 2; row++) {
		for(int col = 0; col < 2; col++) {

			load_dw2_pool_from_DDR(DDR_buff_merge+DW2_POOL_OFFSET, FM_buf1, 0, col, row, col/1, row/1 );
			DW_CONV_3x3_bias(FM_buf1, FM_buf2, weight_buf_3x3[0 + 0], bias_buf[0 + 0], 1);

			load_dw2_pool_from_DDR(DDR_buff_merge+DW2_POOL_OFFSET, FM_buf1, 1, col, row, col/1, row/1 );
			DW_CONV_3x3_bias(FM_buf1, FM_buf3, weight_buf_3x3[0 + 1], bias_buf[0 + 1], 1);

			load_dw2_pool_from_DDR(DDR_buff_merge+DW2_POOL_OFFSET, FM_buf1, 2, col, row, col/1, row/1 );
			DW_CONV_3x3_bias(FM_buf1, FM_buf4, weight_buf_3x3[0 + 2], bias_buf[0 + 2], 1);

			/// DW3_CONV_1x1
			for(int co = 0; co < CO_N; co++) {

				load_bias_from_axi(bias_buf[3], bias_all[bias_1x1_index + co]);

				load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + 0 + co * CI_N]);
				CONV_1x1_bias(FM_buf2, FM_buf_acc, weight_buf_1x1[0], bias_buf[3],0,true);

				load_weight_1x1_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1_index + 1 + co * CI_N]);
				CONV_1x1_bias(FM_buf3, FM_buf_acc, weight_buf_1x1[1], bias_buf[3],0,false);

				load_weight_1x1_from_axi(weight_buf_1x1[2], conv_weight_1x1_all[weight_1x1_index + 2 + co * CI_N]);
				CONV_1x1_bias(FM_buf4, FM_buf_acc, weight_buf_1x1[2], bias_buf[3],0,false);

				relu_copy_buf_to_DDR_acc(DDR_buff_merge+DDR_OFFSET, 100 + co + (col*2 + row) * CO_N, FM_buf_acc, 0, 0);
				Relu_Max_Pooling(FM_buf_acc, DDR_buff_merge, 6 + co, co, col, row, col, row, 3 );
			}
		}
	}


	printf("DW3 Done\n");

	/////////////////////////////// DW4_CONV_3x3 + DW4_CONV_1x1 ////////////////////////////

	/// DW4_CONV_3x3
	// input in DDR_buf[6] - DDR_buf[11]
	// output in DDR_buf[12] - DDR_buf[17]

	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;

	CI_N = 192 / 32;

	/// conv3x3: ping-pong in: FM_buf1 and FM_buf3
	/// conv3x3: out: FM_buf2
	load_buf_from_DDR(FM_buf1, DDR_buff_merge+DDR_OFFSET, 6 + 0);
	for(int c = 0; c < CI_N; c++) {
		load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + c]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c]);

		if( c % 2 == 0 ) {
			load_buf_from_DDR(FM_buf3, DDR_buff_merge+DDR_OFFSET, 6 + c+1);
			DW_CONV_3x3_bias(FM_buf1, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0);
		}
		else {
			load_buf_from_DDR(FM_buf1, DDR_buff_merge+DDR_OFFSET, 6 + c+1);
			DW_CONV_3x3_bias(FM_buf3, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0);
		}

		relu_copy_buf_to_DDR(DDR_buff_merge+DDR_OFFSET, 12 + c, FM_buf2, 0, 0);
	}

	/// DW4_CONV_1x1
	// input in DDR_buf[12] - DDR_buf[17]
	// output in DDR_buf[18] - DDR_buf[29]

	CO_N = 384 / 32;

	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;

	for(int co = 0; co < CO_N; co++) {

		load_bias_from_axi(bias_buf[0], bias_all[bias_1x1_index + co]);

		load_buf_from_DDR(FM_buf1, DDR_buff_merge+DDR_OFFSET, 12 + 0);
		for(int ci = 0; ci < CI_N; ci++) {
			load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);

			if( ci % 2 == 0) {
				load_buf_from_DDR(FM_buf2, DDR_buff_merge+DDR_OFFSET, 12 + ci+1);
				CONV_1x1_bias(FM_buf1, FM_buf_acc, weight_buf_1x1[0],bias_buf[0],0,ci==0);
			}
			else {
				load_buf_from_DDR(FM_buf1, DDR_buff_merge+DDR_OFFSET, 12 + ci+1);
				CONV_1x1_bias(FM_buf2, FM_buf_acc, weight_buf_1x1[0],bias_buf[0],0,ci==0);
			}
		}

		//// Do not write to DDR!! Directly compute DW5_CONV_3x3
		/// DW5_CONV_3x3
		load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + co]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + co]);
		Relu_Convert_FIX(FM_buf_acc, FM_buf3);
		DW_CONV_3x3_bias(FM_buf3, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0);
		relu_copy_buf_to_DDR(DDR_buff_merge+DDR_OFFSET, 30 + co, FM_buf2, 0, 0);

	}


	printf("DW4 Done\n");

	/////////////////////////////// DW5_CONV_3x3 + DW5_CONV_1x1 ////////////////////////////

	/// DW5_CONV_3x3
	// input in DDR_buf[18] - DDR_buf[29]
	// output in DDR_buf[30] - DDR_buf[41]

//	weight_3x3_index += CI_N;
//	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;

	CI_N = 384 / 32;

	/// DW5_CONV_1x1
	// input in DDR_buf[30] - DDR_buf[41]
	// output in DDR_buf[42] - DDR_buf[57]

	CO_N = 512 / 32;

	weight_3x3_index += CI_N;
	bias_3x3_index += CI_N + CO_N;

	for(int co = 0; co < CO_N; co++) {

		load_bias_from_axi(bias_buf[0], bias_all[bias_1x1_index + co]);
		load_buf_from_DDR(FM_buf2, DDR_buff_merge+DDR_OFFSET, 30 + 0);
		for(int ci = 0; ci < CI_N; ci++) {
			load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);

			if( ci % 2 == 0) {
				load_buf_from_DDR(FM_buf1, DDR_buff_merge+DDR_OFFSET, 30 + ci+1);
				CONV_1x1_bias(FM_buf2, FM_buf_acc, weight_buf_1x1[0],bias_buf[0],0,ci==0);
			}
			else {
				load_buf_from_DDR(FM_buf2, DDR_buff_merge+DDR_OFFSET, 30 + ci+1);
				CONV_1x1_bias(FM_buf1, FM_buf_acc, weight_buf_1x1[0],bias_buf[0],0,ci==0);
			}
		}


		relu_copy_buf_to_DDR_acc(DDR_buff_merge+DDR_OFFSET, 42 + co, FM_buf_acc, 0, 0);

		/// compute second half here: don't write back DDR_buf[42] - DDR_buf[57]
		//	// Second half: 768~1280 channels from DW5_conv_1x1_output
		//	// Output of DW5_CONV_1x1_OUT are stored in DDR_buf[42] - DDR_buf[57]
		//	// Output of first half are stored in DDR_buf[82] to DDR_buf[97]
		//  CI_N = 512 / 32;
		load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + co + 24]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + co + 24]);
		Relu_Convert_FIX(FM_buf_acc, FM_buf3);
		DW_CONV_3x3_bias(FM_buf3, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0);
		relu_copy_buf_to_DDR(DDR_buff_merge+DDR_OFFSET, 82 + co, FM_buf2, 0, 0);
	}


	printf("DW5 Done\n");


	/////////////////////////////// CONCAT DW3_CONV_1x1_OUT & DW5_CONV_1x1_OUT /////////////
	/////////////////////////////// DW6_CONV_3x3 + DW6_CONV_1x1 ////////////////////////////

	//// DW6_CONV_3x3
//	weight_3x3_index += CI_N;
//	bias_3x3_index += CI_N + CO_N;
	weight_1x1_index += CO_N * CI_N;
	bias_1x1_index += CO_N + CO_N;


	// First half: 0~767 channels from DW3_conv_1x1_output
	// Output of DW3_CONV_1x1_OUT are stored in DDR_buf[100] to DDR_buf[123]
	// Output of first half are stored in DDR_buf[58] to DDR_buf[81]
	CI_N = 768 / 32;
	for(int c = 0; c < CI_N; c+=4) {

		// From DDR_buf[100] to DDR_buf[123]
		// Img 0: 100 ~ 105
		// Img 1: 106 ~ 111
		// Img 2: 112 ~ 117
		// Img 3: 118 ~ 123

		load_and_reorg(DDR_buff_merge+DDR_OFFSET, 100 + c/4,         DDR_buff_merge+DDR_OFFSET, 100 + c/4 + 6,
					   DDR_buff_merge+DDR_OFFSET, 100 + c/4 + 6 * 2, DDR_buff_merge+DDR_OFFSET, 100 + c/4 + 6 * 3,
                       FM_buf1, FM_buf3, FM_buf4, FM_buf_acc);

		//// 1/4 channels
		load_weight_3x3_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index + c + 0]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c + 0]);
		DW_CONV_3x3_bias(FM_buf1, FM_buf2, weight_buf_3x3[0], bias_buf[0], 0);
		local_buf_copy(FM_buf1, FM_buf_acc);
		relu_copy_buf_to_DDR(DDR_buff_merge+DDR_OFFSET, 58 + c + 0, FM_buf2, 0, 0);

		//// 2/4 channels
		load_weight_3x3_from_axi(weight_buf_3x3[1], conv_weight_3x3_all[weight_3x3_index + c + 1]);
		load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + c + 1]);
		DW_CONV_3x3_bias(FM_buf3, FM_buf2, weight_buf_3x3[1], bias_buf[1], 0);
		relu_copy_buf_to_DDR(DDR_buff_merge+DDR_OFFSET, 58 + c + 1, FM_buf2, 0, 0);

		//// 3/4 channels
		load_weight_3x3_from_axi(weight_buf_3x3[2], conv_weight_3x3_all[weight_3x3_index + c + 2]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c + 2]);
		DW_CONV_3x3_bias(FM_buf4, FM_buf2, weight_buf_3x3[2], bias_buf[0], 0);
		relu_copy_buf_to_DDR(DDR_buff_merge+DDR_OFFSET, 58 + c + 2, FM_buf2, 0, 0);

		//// 4/4 channels
		load_weight_3x3_from_axi(weight_buf_3x3[3], conv_weight_3x3_all[weight_3x3_index + c + 3]);
		load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + c + 3]);
		DW_CONV_3x3_bias(FM_buf1, FM_buf2, weight_buf_3x3[3], bias_buf[1], 0);
		relu_copy_buf_to_DDR(DDR_buff_merge+DDR_OFFSET, 58 + c + 3, FM_buf2, 0, 0);
	}

	/// DW6_CONV_1x1
	// input in DDR_buf[58] - DDR_buf[97]
	// output in DDR_buf[98] - DDR_buf[100]

	bias_1x1_index += 24;
	CO_N = 96 / 32;
	CI_N = 1280 / 32;
	for(int co = 0; co < CO_N; co++) {

		load_bias_from_axi(bias_buf[0], bias_all[bias_1x1_index + co]);

		load_buf_from_DDR(FM_buf2, DDR_buff_merge+DDR_OFFSET, 58 + 0);
		for(int ci = 0; ci < CI_N; ci++) {
			load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);

			if( ci % 2 == 0) {
				load_buf_from_DDR(FM_buf1, DDR_buff_merge+DDR_OFFSET, 58 + ci+1);
				CONV_1x1_bias(FM_buf2, FM_buf_acc, weight_buf_1x1[0],bias_buf[0],0,ci==0);
			}
			else {
				load_buf_from_DDR(FM_buf2, DDR_buff_merge+DDR_OFFSET, 58 + ci+1);
				CONV_1x1_bias(FM_buf1, FM_buf_acc, weight_buf_1x1[0],bias_buf[0],0,ci==0);
			}
		}

		relu_copy_buf_to_DDR_acc(DDR_buff_merge+DDR_OFFSET, 98 + co, FM_buf_acc, 0, 0);
	}


	printf("DW6 Done\n");

	/////////////////////////////// PW7_CONV_1x1 ////////////////////////////
	weight_1x1_index += CO_N * CI_N;
	// input in DDR_buf[98] - DDR_buf[100]
	// output in FM_buf_acc

	CO_N = 32 / 32;
	CI_N = 96 / 32;
	for(int co = 0; co < CO_N; co++) {
		for(int i=0;i<32;i++)
		{
			#pragma HLS unroll
			bias_buf[0][i]=0;
		}
		for(int ci = 0; ci < CI_N; ci++) {
			load_weight_1x1_from_axi(weight_buf_1x1[0], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);
			load_buf_from_DDR(FM_buf2, DDR_buff_merge+DDR_OFFSET, 98 + ci);
			CONV_1x1_bias(FM_buf2, FM_buf_acc, weight_buf_1x1[0],bias_buf[0],0,ci==0);
		}
	}

	printf("PW7 Done\n");

	compute_bounding_box(predict_boxes, constant);

	return;



}

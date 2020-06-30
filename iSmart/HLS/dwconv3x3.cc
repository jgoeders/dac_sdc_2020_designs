

// dw_conv 3x3

#include "net_hls.h"
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"


inline FIX_FM relu_single( FIX_FM d ) {
        if( d > 6 )
                return 6;
        if( d < 0 )
                return 0;
        return d;
}




ap_fixed<24, 7> MAC_16_16(FIX_WT w1, FIX_FM b1, FIX_WT w2, FIX_FM b2)
{
#pragma HLS pipeline

	ap_fixed<24, 7> tmp =  w1 * b1 + w2 * b2;
	return tmp;

}


void DW_CONV_3x3_bias(FIX_FM bottom[32][44][84],
					FIX_FM top[32][44][84],
					FIX_WT weights[32][3][3],
					FIX_WT bias[32],
					int relu
					)
{

#pragma HLS array_partition variable=top dim=1 complete
#pragma HLS array_partition variable=bottom dim=1 complete
#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=bias dim=1 complete


		for(int h = 1; h <= 42; h++){
			for(int w = 1; w <= 82; w++){
#pragma HLS pipeline II=5
				for(int co = 0; co < 32; co++){
#pragma HLS unroll
					ap_fixed<24, 7> tmp1 = MAC_16_16(weights[co][0][0], bottom[co][h-1][w-1], weights[co][0][1], bottom[co][h-1][w  ]);
					ap_fixed<24, 7> tmp2 = MAC_16_16(weights[co][0][2], bottom[co][h-1][w+1], weights[co][1][0], bottom[co][h  ][w-1]);
					ap_fixed<24, 7> tmp3 = MAC_16_16(weights[co][1][1], bottom[co][h  ][w  ], weights[co][1][2], bottom[co][h  ][w+1]);
					ap_fixed<24, 7> tmp4 = MAC_16_16(weights[co][2][0], bottom[co][h+1][w-1], weights[co][2][1], bottom[co][h+1][w  ]);
					ap_fixed<24, 7> tmp5 = MAC_16_16(weights[co][2][2], bottom[co][h+1][w+1], 0, 0);
					ap_fixed<24, 7> sum = tmp1 + tmp2 + tmp3 + tmp4 + tmp5;

					top[co][h][w] = ((FIX_FM)bias[co])+sum;
				}
			}
		}


	if(relu == 1) {
		for(int h = 1; h <= 42; h+=2){
			for(int w = 1; w <= 82; w++){
	#pragma HLS pipeline
				for(int co = 0; co < 32; co++){
					top[co][h  ][w] = relu_single( top[co][h  ][w ]);
					top[co][h+1][w] = relu_single( top[co][h+1][w ]);
				}
			}
		}
	}
}

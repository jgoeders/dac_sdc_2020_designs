#include "SkyNet.h"

static layer config[layer_count] = {
{ "conv0", 320,160,32, 320,160,32, 0,0,0 },  //conv0
{ "conv1", 320,160,32, 320,160,32, 3,1,1 },  //conv1
{ "conv2", 320,160,32, 320,160,64, 1,1,0 },  //conv2
{ "pool1", 320,160,64, 160,80,64,  2,2,0 },  //pool1
{ "conv3", 160,80,64,  160,80,64,  3,1,1 },  //conv3
{ "conv4", 160,80,64,  160,80,96,  1,1,0 },  //conv4
{ "pool2", 160,80,96,  80,40,96,   2,2,0 },  //pool2
{ "conv5", 80,40,96,   80,40,96,   3,1,1 },  //conv5
{ "conv6", 80,40,96,   80,40,192,  1,1,0 },  //conv6
{ "reorg", 80,40,192,  40,20,768,  2,2,0 },  //reorg
{ "pool3", 80,40,192,  40,20,192,  2,2,0 },  //pool3
{ "conv7", 40,20,192,  40,20,192,  3,1,1 },  //conv7
{ "conv8", 40,20,192,  40,20,384,  1,1,0 },  //conv8
{ "conv9", 40,20,384,  40,20,384,  3,1,1 },  //conv9
{ "conv10",40,20,384,  40,20,512,  1,1,0 },  //conv10
{ "cat",   40,20,192,  40,20,1280, 0,0,0 },  //concat
{ "conv11",40,20,1280, 40,20,1280, 3,1,1 },  //conv11
{ "conv12",40,20,1280, 40,20,96,   1,1,0 },  //conv12
{ "conv13",40,20,96,   40,20,32,   1,1,0 },  //conv13
};

ADT FM1[32][43][83];
ADT FM2[32][43][83];
ADT FM3[32][43][83];
BDT FM4[32][43][83];

WDT WBUF3x3[3][32][3][3];
WDT WBUF1x1[2][32][32];
BDT BBUF[3][32];
MDT MBUF[3][32];

void REORG(ADT32 *ifm, ADT IFM[32][43][83], ap_uint<6> Cx, ap_uint<3> Rx)
{
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
    ap_uint<20> ifm_index;
    for (ap_uint<7> h = 1; h <= 41; h++)
    {
        for (ap_uint<7> w = 1; w <= 81; w++)
        {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
            if (h != 21 && w != 41)
            {
                ap_uint<2> bias_h = (h >= 22) + (!Rx[1]);
                ap_uint<2> bias_w = (w >= 42) + (!Rx[0]);
                ap_uint<10> h_ = 2 * h - bias_h;
                ap_uint<10> w_ = 2 * w - bias_w;
                ap_uint<20> ifm_index = Cx * 83 * 163 + h_ * 163 + w_;
                ADT32 DATA = ifm[ifm_index];
                for (ap_uint<7> c = 0; c < 32; c++)
                {
#pragma HLS UNROLL
                    IFM[c][h][w] = DATA.range(8*c+7, 8*c);
                }
            }
        }
    }
}

BDT clamp_BDT(DT x, BDT min, BDT max)
{
#pragma HLS INLINE
    DT y;
    if(x<min) y=min;
    else if(x>max) y=max;
    else y = x;
    return y;
}

void DWCONV3X3(ADT IFM[32][43][83], BDT OFM[32][43][83], WDT WBUF3x3[32][3][3])
{
#pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=WBUF3x3 dim=1 complete
    DT odata = 0;
    for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			for(int h=1; h<42; h++){
				for(int w=1; w<82; w++){
#pragma HLS PIPELINE II=1
					for(int c=0; c<32; c++){
                        odata = OFM[c][h][w];
                        odata += IFM[c][h+i-1][w+j-1]*WBUF3x3[c][i][j];
                        OFM[c][h][w] = clamp_BDT(odata, bmin, bmax);
					}
				}
			}
		}
	}
}

BDT compute_engine_16(WDT w0,  ADT b0,
					  WDT w1,  ADT b1,
					  WDT w2,  ADT b2,
					  WDT w3,  ADT b3,
					  WDT w4,  ADT b4,
					  WDT w5,  ADT b5,
					  WDT w6,  ADT b6,
					  WDT w7,  ADT b7,
					  WDT w8,  ADT b8,
					  WDT w9,  ADT b9,
					  WDT w10, ADT b10,
					  WDT w11, ADT b11,
					  WDT w12, ADT b12,
					  WDT w13, ADT b13,
					  WDT w14, ADT b14,
					  WDT w15, ADT b15)
{
	BDT mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	BDT mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	BDT add0, add1, add2, add3,  add4,  add5,  add6;
	BDT add7, add8, add9, add10, add11, add12, add13, add14;

	mul0  = w0  * b0;
	mul1  = w1  * b1;
	mul2  = w2  * b2;
	mul3  = w3  * b3;
	mul4  = w4  * b4;
	mul5  = w5  * b5;
	mul6  = w6  * b6;
	mul7  = w7  * b7;
	mul8  = w8  * b8;
	mul9  = w9  * b9;
	mul10 = w10 * b10;
	mul11 = w11 * b11;
	mul12 = w12 * b12;
	mul13 = w13 * b13;
	mul14 = w14 * b14;
	mul15 = w15 * b15;

	add0 = mul0  + mul1;
	add1 = mul2  + mul3;
	add2 = mul4  + mul5;
	add3 = mul6  + mul7;
	add4 = mul8  + mul9;
	add5 = mul10 + mul11;
	add6 = mul12 + mul13;
	add7 = mul14 + mul15;

	add8  = add0 + add1;
	add9  = add2 + add3;
	add10 = add4 + add5;
	add11 = add6 + add7;

	add12 = add8  + add9;
	add13 = add10 + add11;

	add14 = add12 + add13;

	return add14;
}

void LOAD_W1x1(WDT WBUF1x1[32][32], WDT W1x1[32][16], int CI)
{
	for(int ci=0; ci<16; ci++){
		for(int co=0; co<32; co++){
#pragma HLS PIPELINE
			W1x1[co][ci] = WBUF1x1[co][ci+CI];
		}
	}
}

void PWCONV1X1(ADT IFM[32][43][83], BDT OFM[32][43][83], WDT WBUF1x1[32][32])
{
#pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=WBUF1x1 dim=1 complete

    WDT W1x1[32][16];
#pragma HLS ARRAY_PARTITION variable=W1x1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=W1x1 dim=2 complete

	for(int ci=0; ci<32; ci+=16)
    {
        LOAD_W1x1(WBUF1x1, W1x1, ci);
		for(int h=1; h<42; h++)
        {
			for(int w=1; w<82; w++)
            {
#pragma HLS PIPELINE II=2
				for(int co=0; co<32; co++)
                {
#pragma HLS UNROLL
                    DT odata = OFM[co][h][w];
					odata += compute_engine_16(
                                    W1x1[co][0],   IFM[ci+0][h][w],
                                    W1x1[co][1],   IFM[ci+1][h][w],
                                    W1x1[co][2],   IFM[ci+2][h][w],
                                    W1x1[co][3],   IFM[ci+3][h][w],
                                    W1x1[co][4],   IFM[ci+4][h][w],
                                    W1x1[co][5],   IFM[ci+5][h][w],
                                    W1x1[co][6],   IFM[ci+6][h][w],
                                    W1x1[co][7],   IFM[ci+7][h][w],
                                    W1x1[co][8],   IFM[ci+8][h][w],
                                    W1x1[co][9],   IFM[ci+9][h][w],
                                    W1x1[co][10],  IFM[ci+10][h][w],
                                    W1x1[co][11],  IFM[ci+11][h][w],
                                    W1x1[co][12],  IFM[ci+12][h][w],
                                    W1x1[co][13],  IFM[ci+13][h][w],
                                    W1x1[co][14],  IFM[ci+14][h][w],
                                    W1x1[co][15],  IFM[ci+15][h][w]);
                    OFM[co][h][w] = clamp_BDT(odata, bmin, bmax);
				}
			}
		}
	}
}

ADT MAX(ADT a, ADT b, ADT c, ADT d)
{
#pragma HLS INLINE
	ADT t1 = a > b ? a : b;
	ADT t2 = c > d ? c : d;
	return t1 > t2 ? t1 : t2;
}

void POOL(ADT IFM[32][43][83], BDT OFM[32][43][83])
{
#pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
	for(int h=1; h<=20; h++){
		for(int w=1; w<=40; w++){
#pragma HLS PIPELINE
			for (int c=0; c<32; c++){
                OFM[c][h][w] = (BDT)MAX(IFM[c][2*h-1][2*w-1],IFM[c][2*h-1][2*w],IFM[c][2*h][2*w-1],IFM[c][2*h][2*w]);
			}
		}
	}
}

ADT clamp_adt(DT x, ADT min, ADT max)
{
#pragma HLS INLINE
    ADT y;
    if(x<min) y=min;
    else if(x>max) y=max;
    else y = x;
    return y;
}

DT ReLU(DT x)
{
#pragma HLS INLINE
    DT y;
    if(x<0) y=0;
    else y=x;
    return y;
}

void ACTIVATION(BDT IFM[32][43][83], ADT OFM[32][43][83], BDT BBUF[32], MDT MBUF[32])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=BBUF dim=1 complete
#pragma HLS ARRAY_PARTITION variable=MBUF dim=1 complete
    DT qy;
	ADT y;
	for(int h=0; h<43; h++){
		for(int w=0; w<83; w++){
#pragma HLS PIPELINE
			for(int c=0; c<32; c++){
                qy = IFM[c][h][w] + BBUF[c];
                qy = ReLU(qy);
                qy = (qy*MBUF[c]) >> nm;
                y = clamp_adt(qy, amin, amax);
                IFM[c][h][w] = 0;
                if(h==0|h==42|w==0|w==82)
                    OFM[c][h][w] = 0;
                else
                    OFM[c][h][w] = y;
			}
		}
	}
}

void Load_WBUF3x3(WDT32* weight, WDT WBUF3x3[32][3][3], int Mx)
{
    for(int m=0; m<3; m++)
    {
        for(int n=0; n<3; n++)
        {
#pragma HLS PIPELINE II=1
            WDT32 DATA;
            DATA = weight[Mx*9+m*3+n];
            for(int c=0; c<32; c++)
            {
                #ifdef __AP_INT__
                WBUF3x3[c][m][n] = DATA.range(8*c+5,8*c);
                #else
                WBUF3x3[c][m][n] = DATA.range(8*c+7,8*c);
                #endif
            }
        }
    }
}

void Load_WBUF1x1(WDT32* weight, WDT WBUF1x1[32][32], int Mx, int Nx, int ic)
{
    for(int n=0; n<32; n++)
    {
#pragma HLS PIPELINE II=1
        WDT32 DATA;
        DATA = weight[Mx*ic+Nx*32+n];
        for(int m=0; m<32; m++)
        {
            #ifdef __AP_INT__
            WBUF1x1[m][n] = DATA.range(8*m+5,8*m);
            #else
            WBUF1x1[m][n] = DATA.range(8*m+7,8*m);
            #endif
        }
    }
}

void Load_BBUF(BDT16* bias, BDT BBUF[32], int Mx)
{
    for(int i=0; i<2; i++)
    {
        BDT16 DATA;
        DATA = bias[Mx*2+i];
#pragma HLS PIPELINE II=1
        for(int c=0; c<16; c++){
            BBUF[i*16+c] = DATA.range(16*c+15, 16*c);
        }
    }
}

void Load_FM(ADT32* ifm, ADT IFM[32][43][83], int Hx, int Wx, int Cx, int ow, int oh)
{
    int tile = ow/80;
    int h_o, w_o;
    if(tile)
    {
        h_o = Hx*40 + Hx/tile;
        w_o = Wx*80 + Wx/tile;
    }
    else
    {
        h_o = 0;
        w_o = 0;
    }
        
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
#pragma HLS PIPELINE II=1
            int ifm_index = Cx*(oh*2+3)*(ow*2+3) + (h+h_o)*(ow*2+3) + (w+w_o);
            ADT32 DATA;
            DATA = ifm[ifm_index];
            for (int c=0; c<32; c++)
            {
                IFM[c][h][w] = DATA.range(8*c+7,8*c);
            }
        }
    }
}

void Export_CONV(ADT32* fm, ADT OFM[32][43][83], int Hx, int Wx, int Cx, int ow, int oh)
{
    int tile = ow/80;
    int h_o, w_o;
    if(tile)
    {
        h_o = Hx*40 + Hx/tile;
        w_o = Wx*80 + Wx/tile;
    }
    else
    {
        h_o = 0;
        w_o = 0;
    }
    for (int h=1; h<=40; h++)
    {
        for (int w=1; w<=80; w++)
        {
#pragma HLS PIPELINE II=1
            int fm_index = Cx*(oh*2+3)*(ow*2+3) + (h+h_o)*(ow*2+3) + (w+w_o);
            ADT32 DATA;
            for (int c=0; c<32; c++)
            {
                DATA.range(8*c+7,8*c) = OFM[c][h][w];
            }
            fm[fm_index] = DATA;
        }
    }
}

void Export_POOL(ADT32* fm, BDT OFM[32][43][83], int Hx, int Wx, int Cx, int ow, int oh)
{
    int tile = ow/40;
    int h_o = Hx*20 + Hx/tile;
    int w_o = Wx*40 + Wx/tile;
    for (int h=1; h<=20; h++)
    {
        for (int w=1; w<=40; w++)
        {
#pragma HLS PIPELINE II=1
            int fm_index = Cx*(oh*2+3)*(ow*2+3) + (h+h_o)*(ow*2+3) + (w+w_o);
            ADT32 DATA;
            for (int c=0; c<32; c++)
            {
                DATA.range(8*c+7,8*c) = OFM[c][h][w];
            }
            fm[fm_index] = DATA;
        }
    }
}

void Load_FM1(ADT32* ifm, ADT IFM[32][43][83], int Cx)
{
    for (int h=0; h<43; h++)
    {
        for (int w=0; w<83; w++)
        {
#pragma HLS PIPELINE II=1
            int ifm_index = Cx*43*83 + h*83 + w;
            ADT32 DATA;
            DATA = ifm[ifm_index];
            for (int c=0; c<32; c++)
            {
                IFM[c][h][w] = DATA.range(8*c+7,8*c);
            }
        }
    }
}

void Export_CONV1(ADT32* fm, ADT OFM[32][43][83], int Cx)
{
    for (int h=0; h<43; h++)
    {
#pragma HLS PIPELINE II=1
        for (int c=0; c<32; c++)
        {
            OFM[c][h][41] = 0;
        }
    }

    for (int w=0; w<83; w++)
    {
#pragma HLS PIPELINE II=1
        for (int c=0; c<32; c++)
        {
            OFM[c][21][w] = 0;
        }
    }

    for (int h=1; h<42; h++)
    {
        for (int w=1; w<82; w++)
        {
#pragma HLS PIPELINE II=1
            int ofm_index = Cx*43*83 + h*83 + w;
            ADT32 DATA;
            for (int c=0; c<32; c++)
            {
                DATA.range(8*c+7,8*c) = OFM[c][h][w];
            }
            fm[ofm_index] = DATA;
        }
    }
}

void CLR_FM(BDT FM[32][43][83])
{
    for(int h=0; h<43; h++)
    {
        for(int w=0; w<83; w++)
        {
#pragma HLS PIPELINE II=1
            for(int c=0; c<32; c++)
            {
                FM[c][h][w] = 0;
            }
        }
    }
}

void Export_BBOX(BDT16* bbox, BDT16 BBOX[4])
{
    for (int i=0; i<4; i++)
    {
#pragma HLS PIPELINE II=1
        bbox[i] = BBOX[i];
    }
}

void Compute_BBOX(BDT OFM[32][43][83], BDT MBUF[32], BDT16 BBOX[4])
{
    int H,W;
    DT conf[2];
    BDT max[2], h_max[2], w_max[2];
    BDT xs[4], ys[4], ws[4], hs[4], flag[4], x[4], y[4];

    for(int b=0; b<4; b++)
    {
        switch(b)
        {
            case 0: H=1; W=1; break;
            case 1: H=1; W=42; break;
            case 2: H=22; W=1; break;
            case 3: H=22; W=42; break;
        }
        max[0] = OFM[4][H][W];
        h_max[0] = H;
        w_max[0] = W;
        max[1] = OFM[9][H][W];
        h_max[1] = H;
        w_max[1] = W;

        for(int h=0; h<20; h++){
            for(int w=0; w<40; w++){
#pragma HLS PIPELINE II=1
                if(OFM[4][h+H][w+W]>max[0]){
                    max[0] = OFM[4][h+H][w+W];
                    h_max[0] = h+H;
                    w_max[0] = w+W;
                }
                if(OFM[9][h+H][w+W]>max[1]){
                    max[1] = OFM[9][h+H][w+W];
                    h_max[1] = h+H;
                    w_max[1] = w+W;
                }
            }
        }
        conf[0] = max[0]*MBUF[4];
        conf[1] = max[1]*MBUF[9];
        if(conf[1]>conf[0])
        {
            xs[b] = OFM[5][h_max[1]][w_max[1]];
            ys[b] = OFM[6][h_max[1]][w_max[1]];
            ws[b] = OFM[7][h_max[1]][w_max[1]];
            hs[b] = OFM[8][h_max[1]][w_max[1]];
            flag[b] = 1;
            x[b] = w_max[1]-W;
            y[b] = h_max[1]-H;
        }
        else
        {
            xs[b] = OFM[0][h_max[0]][w_max[0]];
            ys[b] = OFM[1][h_max[0]][w_max[0]];
            ws[b] = OFM[2][h_max[0]][w_max[0]];
            hs[b] = OFM[3][h_max[0]][w_max[0]];
            flag[b] = 0;
            x[b] = w_max[0]-W;
            y[b] = h_max[0]-H;
        }
        BBOX[b].range(15,0)  = xs[b];
        BBOX[b].range(31,16) = ys[b];
        BBOX[b].range(47,32) = ws[b];
        BBOX[b].range(63,48) = hs[b];
        BBOX[b].range(79,64) = flag[b];
        BBOX[b].range(95,80) = x[b];
        BBOX[b].range(112,96) = y[b];
    }
}

void Load_IMG(ADT4* img, ADT IFM[32][43][83], int Hx, int Wx, int b)
{
    int h_o = Hx*40-1;
    int w_o = Wx*80-1;
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
#pragma HLS PIPELINE II=1
            ADT4 DATA = img[b*320*160 + (h+h_o)*320 + (w+w_o)];
            for (int c=0; c<3; c++)
            {
                if (h+h_o<0||w+w_o<0||h+h_o>159||w+w_o>319)
                    IFM[c][h][w] = 128;
                else
                    IFM[c][h][w] = DATA.range(8*c+7,8*c);
            }
        }
    }
}

void SkyNet(ADT4* img, ADT32* fm, WDT32* weight, BDT16* biasm)
{
#pragma HLS INTERFACE m_axi depth=204800 port=img    offset=slave bundle=fm
#pragma HLS INTERFACE m_axi depth=628115 port=fm     offset=slave bundle=fm
#pragma HLS INTERFACE m_axi depth=13792  port=weight offset=slave bundle=wt
#pragma HLS INTERFACE m_axi depth=432    port=biasm  offset=slave bundle=bm
#pragma HLS INTERFACE s_axilite register port=return

#pragma HLS ALLOCATION instances=PWCONV1x1		limit=1 function
#pragma HLS ALLOCATION instances=DWCONV3x3	   	limit=1 function
#pragma HLS ALLOCATION instances=REORG	    	limit=1 function
#pragma HLS ALLOCATION instances=POOL	    	limit=1 function
#pragma HLS ALLOCATION instances=ACTIVATION    	limit=1 function
#pragma HLS ALLOCATION instances=Load_FM    	limit=1 function
#pragma HLS ALLOCATION instances=Export_CONV    limit=1 function
#pragma HLS ALLOCATION instances=Load_FM1    	limit=1 function
#pragma HLS ALLOCATION instances=Export_CONV1   limit=1 function

    /*********************************DWCONV1+PWCONV1********************************/
    std::cout << "DWCONV1+PWCONV1" << std::endl;
    Load_WBUF3x3(weight + conv1_w, WBUF3x3[0], 0);
    Load_BBUF(biasm + conv1_b, BBUF[0], 0);
    Load_BBUF(biasm + conv1_m, MBUF[0], 0);

    Load_WBUF1x1(weight + conv2_w, WBUF1x1[0], 0, 0, config[2].ic);
    Load_WBUF1x1(weight + conv2_w, WBUF1x1[1], 1, 0, config[2].ic);
    Load_BBUF(biasm + conv2_b, BBUF[1], 0);
    Load_BBUF(biasm + conv2_b, BBUF[2], 1);
    Load_BBUF(biasm + conv2_m, MBUF[1], 0);
    Load_BBUF(biasm + conv2_m, MBUF[2], 1);

    {
        for(int b=0; b<4; b++)
        {
            int H, W;
            switch(b)
            {
                case 0: H=0; W=0; break;
                case 1: H=0; W=4; break;
                case 2: H=4; W=0; break;
                case 3: H=4; W=4; break;
            }
            for(int Hx=0; Hx<4; Hx++)
            {
                Load_IMG(img, FM1, Hx, 0, b);
                for(int Wx=0; Wx<4; Wx++)
                {
                    if(Wx%2==0)
                    {
                        Load_IMG(img, FM2, Hx, Wx+1, b);
                        DWCONV3X3(FM1, FM4, WBUF3x3[0]);
                        ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                        for(int Mx=0; Mx<2; Mx++)
                        {
                            PWCONV1X1(FM1, FM4, WBUF1x1[Mx]);
                            ACTIVATION(FM4, FM3, BBUF[1+Mx], MBUF[1+Mx]);
                            POOL(FM3, FM4);
                            Export_POOL(fm + pool1_o, FM4, Hx+H, Wx+W, Mx, config[3].ow, config[3].oh);
                            CLR_FM(FM4);
                        }
                    }
                    else
                    {
                        Load_IMG(img, FM1, Hx, Wx+1, b);
                        DWCONV3X3(FM2, FM4, WBUF3x3[0]);
                        ACTIVATION(FM4, FM2, BBUF[0], MBUF[0]);
                        for(int Mx=0; Mx<2; Mx++)
                        {
                            PWCONV1X1(FM2, FM4, WBUF1x1[Mx]);
                            ACTIVATION(FM4, FM3, BBUF[1+Mx], MBUF[1+Mx]);
                            POOL(FM3, FM4);
                            Export_POOL(fm + pool1_o, FM4, Hx+H, Wx+W, Mx, config[3].ow, config[3].oh);
                            CLR_FM(FM4);
                        }
                    }
                }
            }
        }
    }
    /*********************************DWCONV2+PWCONV2********************************/
    std::cout << "DWCONV2+PWCONV2" << std::endl;
    Load_WBUF3x3(weight + conv3_w, WBUF3x3[0], 0);
    Load_WBUF3x3(weight + conv3_w, WBUF3x3[1], 1);
    Load_BBUF(biasm + conv3_b, BBUF[0], 0);
    Load_BBUF(biasm + conv3_b, BBUF[1], 1);
    Load_BBUF(biasm + conv3_m, MBUF[0], 0);
    Load_BBUF(biasm + conv3_m, MBUF[1], 1);
    {
        for(int Hx=0; Hx<4; Hx++)
        {
            for(int Wx=0; Wx<4; Wx++)
            {
                Load_FM(fm + pool1_o, FM1, Hx, Wx, 0, config[4].ow, config[4].oh);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
                ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                
                Load_FM(fm + pool1_o, FM2, Hx, Wx, 1, config[4].ow, config[4].oh);
                DWCONV3X3(FM2, FM4, WBUF3x3[1]);
                ACTIVATION(FM4, FM2, BBUF[1], MBUF[1]);
                
                for(int Mx=0; Mx<3; Mx++)
                {
                    Load_WBUF1x1(weight + conv4_w, WBUF1x1[0], Mx, 0, config[5].ic);
                    PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                    Load_WBUF1x1(weight + conv4_w, WBUF1x1[1], Mx, 1, config[5].ic);
                    PWCONV1X1(FM2, FM4, WBUF1x1[1]);

                    Load_BBUF(biasm + conv4_b, BBUF[2], Mx);
                    Load_BBUF(biasm + conv4_m, MBUF[2], Mx);
                    ACTIVATION(FM4, FM3, BBUF[2], MBUF[2]);
                    POOL(FM3, FM4);
                    Export_POOL(fm + pool2_o, FM4, Hx, Wx, Mx, config[6].ow, config[6].oh);
                    CLR_FM(FM4);
                }
            }
        }
    }
    /*********************************DWCONV3+PWCONV3********************************/
    std::cout << "DWCONV3+PWCONV3" << std::endl;
    Load_WBUF3x3(weight + conv5_w, WBUF3x3[0], 0);
    Load_WBUF3x3(weight + conv5_w, WBUF3x3[1], 1);
    Load_WBUF3x3(weight + conv5_w, WBUF3x3[2], 2);
    Load_BBUF(biasm + conv5_b, BBUF[0], 0);
    Load_BBUF(biasm + conv5_b, BBUF[1], 1);
    Load_BBUF(biasm + conv5_b, BBUF[2], 2);
    Load_BBUF(biasm + conv5_m, MBUF[0], 0);
    Load_BBUF(biasm + conv5_m, MBUF[1], 1);
    Load_BBUF(biasm + conv5_m, MBUF[2], 2);
    {
        for(int Hx=0; Hx<2; Hx++)
        {
            for(int Wx=0; Wx<2; Wx++)
            {
                Load_FM(fm + pool2_o, FM1, Hx, Wx, 0, config[7].ow, config[7].oh);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
                ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                Export_CONV(fm + conv5_o, FM1, Hx, Wx, 0, config[7].ow, config[7].oh);

                Load_FM(fm + pool2_o, FM1, Hx, Wx, 1, config[7].ow, config[7].oh);
                DWCONV3X3(FM1, FM4, WBUF3x3[1]);
                ACTIVATION(FM4, FM1, BBUF[1], MBUF[1]);
                Export_CONV(fm + conv5_o, FM1, Hx, Wx, 1, config[7].ow, config[7].oh);

                Load_FM(fm + pool2_o, FM1, Hx, Wx, 2, config[7].ow, config[7].oh);
                DWCONV3X3(FM1, FM4, WBUF3x3[2]);
                ACTIVATION(FM4, FM1, BBUF[2], MBUF[2]);
                Export_CONV(fm + conv5_o, FM1, Hx, Wx, 2, config[7].ow, config[7].oh);
            }
        }
    }
    
    {
        for(int Hx=0; Hx<2; Hx++)
        {
            for(int Wx=0; Wx<2; Wx++)
            {
                for(int Mx=0; Mx<6; Mx++)
                {   
                    for(int Nx=0; Nx<3; Nx++)
                    {
                        Load_FM(fm + conv5_o, FM1, Hx, Wx, Nx, config[7].ow, config[7].oh);
                        Load_WBUF1x1(weight + conv6_w, WBUF1x1[1], Mx, Nx, config[8].ic);
                        PWCONV1X1(FM1, FM4, WBUF1x1[1]);
                    }
                    Load_BBUF(biasm + conv6_b, BBUF[0], Mx);
                    Load_BBUF(biasm + conv6_m, MBUF[0], Mx);
                    ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                    Export_CONV(fm + conv6_o, FM1, Hx, Wx, Mx, config[8].ow, config[8].oh);
                    POOL(FM1, FM4);
                    Export_POOL(fm + pool3_o, FM4, Hx, Wx, Mx, config[10].ow, config[10].oh);
                    CLR_FM(FM4);
                }
            }
        }
    }

    /*********************************DWCONV4+PWCONV4********************************/
    std::cout << "DWCONV4+PWCONV4" << std::endl;
    {
        Load_FM1(fm + pool3_o, FM1, 0);
        for(int Nx=0; Nx<6; Nx++)
        {
            Load_WBUF3x3(weight + conv7_w, WBUF3x3[0], Nx);
            Load_BBUF(biasm + conv7_b, BBUF[0], Nx);
            Load_BBUF(biasm + conv7_m, MBUF[0], Nx);
            if(Nx%2==0)
            {
                Load_FM1(fm + pool3_o, FM2, Nx+1);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
            }
            else
            {
                Load_FM1(fm + pool3_o, FM1, Nx+1);
                DWCONV3X3(FM2, FM4, WBUF3x3[0]);
            }
            ACTIVATION(FM4, FM3, BBUF[0], MBUF[0]);
            Export_CONV1(fm + conv7_o, FM3, Nx);
        }
    }

    {
        for(int Mx=0; Mx<12; Mx++)
        {
            Load_BBUF(biasm + conv8_b, BBUF[0], Mx);
            Load_BBUF(biasm + conv8_m, MBUF[0], Mx);
            Load_FM1(fm + conv7_o, FM1, 0);
            for(int Nx=0; Nx<6; Nx++)
            {
                Load_WBUF1x1(weight + conv8_w, WBUF1x1[0], Mx, Nx, config[12].ic);
                if(Nx%2==0)
                {
                    Load_FM1(fm + conv7_o, FM2, Nx+1);
                    PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                }
                else
                {
                    Load_FM1(fm + conv7_o, FM1, Nx+1);
                    PWCONV1X1(FM2, FM4, WBUF1x1[0]);
                }
            }
            ACTIVATION(FM4, FM2, BBUF[0], MBUF[0]);
            Export_CONV1(fm + conv8_o, FM2, Mx);
        }
    }

    /*********************************DWCONV5+PWCONV5********************************/
    std::cout << "DWCONV5+PWCONV5" << std::endl;
    {
        Load_FM1(fm + conv8_o, FM1, 0);
        for(int Nx=0; Nx<12; Nx++)
        {
            Load_WBUF3x3(weight + conv9_w, WBUF3x3[0], Nx);
            Load_BBUF(biasm + conv9_b, BBUF[0], Nx);
            Load_BBUF(biasm + conv9_m, MBUF[0], Nx);
            if(Nx%2==0)
            {
                Load_FM1(fm + conv8_o, FM2, Nx+1);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
            }
            else
            {
                Load_FM1(fm + conv8_o, FM1, Nx+1);
                DWCONV3X3(FM2, FM4, WBUF3x3[0]);
            }
            ACTIVATION(FM4, FM3, BBUF[0], MBUF[0]);
            Export_CONV1(fm + conv9_o, FM3, Nx);
        }
    }
    {
        Load_FM1(fm + conv9_o, FM1, 0);
        for(int Mx=0; Mx<16; Mx++)
        {
            Load_BBUF(biasm + conv10_b, BBUF[0], Mx);
            Load_BBUF(biasm + conv10_m, MBUF[0], Mx);
            Load_FM1(fm + conv9_o, FM1, 0);
            for(int Nx=0; Nx<12; Nx++)
            {
                Load_WBUF1x1(weight + conv10_w, WBUF1x1[0], Mx, Nx, config[14].ic);
                if(Nx%2==0)
                {
                    Load_FM1(fm + conv9_o, FM2, Nx+1);
                    PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                }
                else
                {
                    Load_FM1(fm + conv9_o, FM1, Nx+1);
                    PWCONV1X1(FM2, FM4, WBUF1x1[0]);
                }
            }
            ACTIVATION(FM4, FM2, BBUF[0], MBUF[0]);
            Export_CONV1(fm + conv10_o, FM2, Mx);
        }
    }
    /*********************************REORG+CONCAT+DWCONV6********************************/
    std::cout << "REORG+DWCONV6" << std::endl;
    {
        for(int Nx=0; Nx<6; Nx++)
        {
            for(int Rx=0; Rx<4; Rx++)
            {
                REORG(fm + conv6_o, FM1, Nx, Rx);
                Load_WBUF3x3(weight + conv11_w, WBUF3x3[0], Nx+6*Rx);
                Load_BBUF(biasm + conv11_b, BBUF[0], Nx+6*Rx);
                Load_BBUF(biasm + conv11_m, MBUF[0], Nx+6*Rx);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
                ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                Export_CONV1(fm + conv11_o, FM1, Nx+6*Rx);
            }
        }
    }

    {
        Load_FM1(fm + conv10_o, FM1, 0);
        for(int Nx=0; Nx<16; Nx++)
        {
            Load_WBUF3x3(weight + conv11_w, WBUF3x3[0], Nx+24);
            Load_BBUF(biasm + conv11_b, BBUF[0], Nx+24);
            Load_BBUF(biasm + conv11_m, MBUF[0], Nx+24);
            if(Nx%2==0)
            {
                Load_FM1(fm + conv10_o, FM2, Nx+1);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
            }
            else
            {
                Load_FM1(fm + conv10_o, FM1, Nx+1);
                DWCONV3X3(FM2, FM4, WBUF3x3[0]);
            }
            ACTIVATION(FM4, FM3, BBUF[0], MBUF[0]);
            Export_CONV1(fm + conv11_o, FM3, Nx+24);
        }
    }
    /*********************************PWCONV6********************************/
    std::cout << "PWCONV6" << std::endl;
    {
        for(int Mx=0; Mx<3; Mx++)
        {
            Load_FM1(fm + conv11_o, FM1, 0);
            for(int Nx=0; Nx<40; Nx++)
            {
                Load_WBUF1x1(weight + conv12_w, WBUF1x1[0], Mx, Nx, config[17].ic);
                if(Nx%2==0)
                {
                    Load_FM1(fm + conv11_o, FM2, Nx+1);
                    PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                }
                else
                {
                    Load_FM1(fm + conv11_o, FM1, Nx+1);
                    PWCONV1X1(FM2, FM4, WBUF1x1[0]);
                }
            }
            Load_BBUF(biasm + conv12_b, BBUF[0], Mx);
            Load_BBUF(biasm + conv12_m, MBUF[0], Mx);
            ACTIVATION(FM4, FM3, BBUF[0], MBUF[0]);
            Export_CONV1(fm + conv12_o, FM3, Mx);
        }
    }

    /*********************************CONV13********************************/
    std::cout << "CONV13" << std::endl;
    for(int Nx=0; Nx<3; Nx++)
    {
        Load_FM1(fm + conv12_o, FM1, Nx);
        Load_WBUF1x1(weight + conv13_w, WBUF1x1[0], 0, Nx, config[18].ic);
        PWCONV1X1(FM1, FM4, WBUF1x1[0]);
    }
    Load_BBUF(biasm + conv13_m, MBUF[0], 0);
    BDT16 BBOX[4];
    Compute_BBOX(FM4, MBUF[0], BBOX);
    Export_BBOX(biasm + bbox_o, BBOX);
    CLR_FM(FM4);
}

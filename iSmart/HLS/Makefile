AUTOPILOT_ROOT :=/opt/xilinx/sdsoc/2018.2/Vivado/2018.2/

ASSEMBLE_SRC_ROOT := .
TB_ROOT := .
IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}"
IFLAG += -I "${ASSEMBLE_SRC_ROOT}" 
IFLAG += -I "/usr/include/x86_64-linux-gnu"
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__

# IFLAG += -DDEBUG_FILE_PRINT=1
IFLAG +=  -g 
CFLAG += -fPIC  -O3 #-fsanitize=address
CC      = g++ 


ALLOUT+= csim.out

all: $(ALLOUT) 
##TO BE MODIFIED START

conv1x1.o:./conv1x1.cc
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
dwconv3x3.o:./dwconv3x3.cc
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
golden_c.o:./golden_c.cc
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
net_hls.o:./net_hls.cc
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
output_verify.o:./output_verify.cc 
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
reorder_weight.o:./reorder_weight.cc 
	$(CC) $(GCOV)  $(CFLAG)  -o $@ -c $^    -MMD $(IFLAG)
##TO BE MODIFIED END



IP_DEP+=conv1x1.o
IP_DEP+=dwconv3x3.o
IP_DEP+=golden_c.o
IP_DEP+=net_hls.o
IP_DEP+=output_verify.o
IP_DEP+=reorder_weight.o

main.o:./tb.cc
	$(CC) $(GCOV)  $(CFLAG)  -I "${ASSEMBLE_SRC_ROOT}" -o $@  -c $^   -MMD $(IFLAG)

csim.out: main.o $(IP_DEP)
	$(CC)  $(GCOV)  $(CFLAG) -MMD $(IFLAG)  -o $@  $^ 

clean:
	rm -f -r csim.d 
	rm -f *.out *.gcno *.gcda *.txt *.o *.d





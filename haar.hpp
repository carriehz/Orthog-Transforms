*
 * haar.hpp
 *
 *  Created on: Mar 18, 2014
 *      Author: Carrie Hernandez
 *
 *      Haar transform and inverse implementation for nxn sized images. Includes Haar wavelet with iterations and
 *      Kronecker Product functions
 */

#ifndef HAAR_H_
#define HAAR_H_


using namespace cv;
using namespace std;

/********************************************************************/
void haar(float, InputArray, OutputArray); // Haar transform
void ihaar(float, InputArray, OutputArray); //inverse Haar
void haar_w(float, InputArray, OutputArray); //Haar wavelet iterations
//void kron_prod( InputArray, InputArray, OutputArray); // Kronecker Product

/*********************************************************************/
void haar(float n, InputArray _src, OutputArray _dst){

	Mat src = _src.getMat();
	float ncol = src.cols;
	float nrow = src.rows;

	_dst.create(nrow, ncol, CV_32F);
	Mat dst = _dst.getMat();

 float N = pow(2.0f, n);
 float had2[2][2] = { 1,  1,  1,  -1};
 float h2a[1][2] = {1, 1};
 float h2b[1][2] = {1, -1};
 Mat H2a = Mat(1, 2, CV_32F, h2a);
 Mat H2b = Mat(1, 2, CV_32F, h2b);
 Mat H2 = Mat(2, 2, CV_32F, had2);
 Mat Haar = Mat(nrow, ncol, CV_32F);
 Mat H, Ht, Haart, Htemp1, Htemp2;
 Mat sI;
 float s, p2n1;

 for ( int i = 2; i <= n; i++ ){
	  s = sqrt(pow(2.0f, (i-1)));
	  p2n1 =  pow(2.0f, (i-1));
	  Mat I = Mat::eye(p2n1, p2n1, CV_32F);
	  sI = s*I;
	 if ( i == 2){
		  kron_prod(H2, H2a, Htemp1);
		  kron_prod(sI, H2b, Htemp2);
		  vconcat(Htemp1, Htemp2, H);
	 }
	 else {

		 kron_prod(H, H2a, Htemp1);
		 kron_prod(sI, H2b, Htemp2);
		 vconcat(Htemp1, Htemp2, H);
	 }
 }
 	 Haar = ( 1/sqrt(N)) * H;
	transpose(Haar, Haart);
	dst = (Haar*src)*Haart; //Haar Transform

}
void ihaar(float n, InputArray _src, OutputArray _dst){

 float N = pow(2.0f, n);

 float had2[2][2] = { 1,  1,  1,  -1};
 float h2a[1][2] = {1, 1};
 float h2b[1][2] = {1, -1};
 Mat H2a = Mat(1, 2, CV_32F, h2a);
 Mat H2b = Mat(1, 2, CV_32F, h2b);
 Mat H2 = Mat(2, 2, CV_32F, had2);
 Mat Haar = Mat(N, N, CV_32F);
 Mat Haart, Htemp1, Htemp2;
 Mat sI;

 float s, p2n1;

 for ( int i = 2; i <= n; i++ ){

	  s = sqrt(pow(2.0f, (i-1)));
	  p2n1 =  pow(2.0f, (i-1));
	  Mat I = Mat::eye(p2n1, p2n1, CV_32F);
	  sI = s*I;

	 if ( i == 2){

		  kron_prod(H2, H2a, Htemp1);
		  kron_prod(sI, H2b, Htemp2);
		  vconcat(Htemp1, Htemp2, Haar);

	 }

	 else {

		 kron_prod(Haar, H2a, Htemp1);
		 kron_prod(sI, H2b, Htemp2);
		 vconcat(Htemp1, Htemp2, Haar);
	 }

 }

 	 Haar = ( 1/sqrt(N)) * Haar;

	Mat src = _src.getMat();
	float ncol = src.cols;
	float nrow = src.rows;
	_dst.create(nrow, ncol, CV_32F);
	Mat dst = _dst.getMat();

	Mat temp, U;
	Mat out = Mat(256, 256, CV_32F);
	Scalar S;

	transpose(Haar, Haart);

	dst = (Haart*src)*Haar;

}

void haar_w(float n, InputArray _src, OutputArray _dst){

		Mat src = _src.getMat();
		float ncol = src.cols;
		float nrow = src.rows;

		_dst.create(nrow, ncol, CV_32F);
		Mat dst = _dst.getMat();
	 float N = pow(2.0f, n);
	 float had2[2][2] = { 1,  1,  1,  -1};
	 float h2a[1][2] = {1, 1};
	 float h2b[1][2] = {1, -1};
	 Mat H2a = Mat(1, 2, CV_32F, h2a);
	 Mat H2b = Mat(1, 2, CV_32F, h2b);
	 Mat H2 = Mat(2, 2, CV_32F, had2);
	 Mat Haar = Mat(nrow, ncol, CV_32F);

	 Mat H, Ht, Haart, Htemp1, Htemp2;
	 Mat sI;

	 float s, p2n1;

	 for ( int i = 2; i <= n; i++ ){

		  s = sqrt(pow(2.0f, (i-1)));
		  p2n1 =  pow(2.0f, (i-1));
		  Mat I = Mat::eye(p2n1, p2n1, CV_32F);
		  sI = s*I;
		 if ( i == 2){
			  kron_prod(H2, H2a, Htemp1);
			  kron_prod(sI, H2b, Htemp2);
			  vconcat(Htemp1, Htemp2, H);
		 }
		 else {

			 kron_prod(H, H2a, Htemp1);
			 kron_prod(sI, H2b, Htemp2);
			 vconcat(Htemp1, Htemp2, H);
		 }
	 }
	 	Haar = ( 1/sqrt(N)) * H;
		transpose(Haar, Haart);

//Haar Wavelet Transform

	Mat out, out1, out2, temp, UL, res, sout1;
	Mat low, high, lowt, hight, LH, LHt, slow, shigh, sLH, sLHt;

	low = (sqrt(2)/2)*abs((Haar.rowRange(128, 256)));
	high = -4*(Haar.rowRange(128, 256));

	vconcat(low, high, LH);
	transpose(LH, LHt);
	out1 = (LH*src);
	out = (out1*LHt);
	out.copyTo(dst);
	out.copyTo(temp);

	low = abs((Haar.rowRange(128, 256)));
	high = (Haar.rowRange(128, 256));

	resize(low, slow, Size(), 0.5, 0.5, INTER_CUBIC);//decimate by 2
	resize(high, shigh, Size(), 0.5, 0.5, INTER_CUBIC);//decimate by 2
	resize(temp, res, Size(), 0.5, 0.5, INTER_CUBIC);//decimate by 2

	vconcat(slow, shigh, sLH);
	transpose(sLH, sLHt);

	sout1 = (sLH*res)*sLHt;
	UL = sout1(Range(0, 64), Range(0, 64));
	resize(UL, UL, Size(), 2, 2, INTER_CUBIC);
	UL.copyTo(out(Range(0, 128), Range(0, 128)));
	out = 2*out;
	out.copyTo(dst);

imwrite("/Users/me/cuda-workspace/Wavelets_Project/haar_iterate.bmp", out);
}
/*
void kron_prod( InputArray _src1, InputArray _src2, OutputArray _dst){

	Mat A = _src1.getMat();
	Mat B = _src2.getMat();

	int rsz, csz;

	rsz = A.rows*B.rows;
	csz = A.cols*B.cols;

	Mat out = Mat(rsz, csz, CV_32F);
	_dst.create(rsz, csz, CV_32F);
	Mat dst = _dst.getMat();

	Mat temp = Mat::zeros(B.rows,B.cols, CV_32F);
	Mat res1 = Mat::zeros(B.rows,B.cols, CV_32F);
	Mat res2 = Mat::zeros(rsz, csz, CV_32F);
	Mat res3 = Mat::zeros(rsz, csz, CV_32F);

	for( int i = 0; i < A.rows; i++ ){

		Mat res1 = Mat::zeros(B.rows,B.cols, CV_32F);

		for( int j = 0; j < A.cols; j++ ){

			if (j == 0){
				res1 = A.at<float>(i,j)*B;
				res1.copyTo(out);
			}
			else{
				temp = A.at<float>(i,j)*B;
				hconcat(res1, temp, res1);
				res1.copyTo(res2);
			}
		}

		if (i == 0){
			res2.copyTo(res3);
		}
		else{
			vconcat(res3, res2, res3);
			res3.copyTo(out);
		}
	}
	out.copyTo(dst);
}

*/
#endif /* HAAR_H_ */


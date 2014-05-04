/*
 * fast_walsh.hpp
 *  Created on: Feb 16, 2014
 *      Author: Carrie Hernandez
 *      Fast Walsh Hadamard, Fast Walsh Paley Transforms for 256 x 256 matrices or 8 x 8 pixel blocks.
 */

#ifndef FAST_WALSH_HPP_
#define FAST_WALSH_HPP_

using namespace cv;
using namespace std;

void fast_walsh256( InputArray, OutputArray ); // Fast Walsh Hadamard Transform for 256 x 256 pixel images
void fast_walsh8( InputArray, OutputArray ); // Fast Walsh Hadamard Transform for 8 x 8 pixel blocks
void fast_walsh_paley8( InputArray, OutputArray ); // Fast Walsh Paley Transform for 8 x 8 pixel blocks
void ifast_walsh256( InputArray, OutputArray ); // Inverse Fast Walsh Hadamard Transform for 256 x 256 images
//void kron_prod( InputArray, InputArray, OutputArray); // Kronecker Product function

/************************************************************************/

void fast_walsh256(InputArray _src, OutputArray _dst ){
		Mat src = _src.getMat();
		float ncol = src.cols;
		float nrow = src.rows;

		_dst.create(nrow, ncol, CV_32F);
		Mat dst = _dst.getMat();

		Mat temp;
		Mat out = Mat(256, 256, CV_32F);
		Mat U;
		Scalar S;

		float had2[2][2] = { 1,  1,  1,  -1};
		Mat H2 = Mat(2, 2, CV_32F, had2);
		Mat H256 = Mat(256, 256, CV_32F);
		Mat H256t;

		Mat I2 = Mat::eye(2, 2, CV_32F);
		Mat I4 = Mat::eye(4, 4, CV_32F);
		Mat I8 = Mat::eye(8, 8, CV_32F);
		Mat I16 = Mat::eye(16, 16, CV_32F);
		Mat I32 = Mat::eye(32, 32, CV_32F);
		Mat I64 = Mat::eye(64, 64, CV_32F);
		Mat I128 = Mat::eye(128, 128, CV_32F);

		Mat B1, B2, B3, B4, B5, B6, B7, B8;
		Mat b;
		float s = (1/sqrt(256*256));

//256 point

		kron_prod(H2, I128, B1); //256x256

		kron_prod(H2, I64, b); //128x128
		kron_prod(I2, b, B2); //256x256

		kron_prod(H2, I32, b); //64x64
		kron_prod(I4, b, B3); //256x256

		kron_prod(H2, I16, b); //32x32
		kron_prod(I8, b, B4); //256x256

		kron_prod(H2, I8, b); //16x16
		kron_prod(I16, b, B5); //256x256

		kron_prod(H2, I4, b); //8x8
		kron_prod(I32, b, B6); //256x256

		kron_prod(H2, I2, b); //4x4
		kron_prod(I64, b, B7); //256x256

		kron_prod(I128, H2, B8); //256x256

		H256 = B1*B2*B3*B4*B5*B6*B7*B8;

		transpose(H256, H256t);

		for ( int i = 0; i < ncol; i++ ){
			  for (int j = 0; j < nrow; j++ ){
			  		temp = H256t.col(i) * H256.row(j);
			  		transpose(temp, temp); //adjust for unequal matrices
			  		multiply(temp, src, U);
			  		S = sum(U);
			  		dst.at<float>(i,j) = s*S[0];
			  	}
		}

//16 point
/*		Mat H16;
		kron_prod(H2, I8, B1); //16x16
		kron_prod(H2, I4, b0); //8x8
		kron_prod(I2, b0, B2); //16x16
		kron_prod(H2, I2, b0); //4x4
		kron_prod(I4, b0, B3); //16x16
		kron_prod(I8, H2, B4); //16x16

		H16 = B1*B2*B3*B4;
		std::cout << H16;
*/


}

void fast_walsh8(InputArray _src, OutputArray _dst ){
		Mat src = _src.getMat();
		float ncol = src.cols;
		float nrow = src.rows;
		_dst.create(nrow, ncol, CV_32F);
		Mat dst = _dst.getMat();
		Mat temp;
		Mat out = Mat(8, 8, CV_32F);
		Mat U;
		Scalar S;

		float had2[2][2] = { 1,  1,  1,  -1};
		Mat H2 = Mat(2, 2, CV_32F, had2);
		Mat H8 = Mat(8, 8, CV_32F);
		Mat H8t;

		Mat I2 = Mat::eye(2, 2, CV_32F);
		Mat I4 = Mat::eye(4, 4, CV_32F);

		Mat B1, B2, B3;
		Mat b;
		float s = (1/sqrt(8*8));

//8 point

		kron_prod(H2, I4, B1); //8x8
		kron_prod(H2, I2, b); //4x4
		kron_prod(I2, b, B2); //8x8
		kron_prod(I4, H2, B3); //8x8

		H8 = B1*B2*B3;

		transpose(H8, H8t);

		for ( int i = 0; i < ncol; i++ ){
			  for (int j = 0; j < nrow; j++ ){
			  		temp = H8t.col(i) * H8.row(j);
			  		//transpose(temp, temp); //adjust for unequal matrices
			  		multiply(temp, src, U);
			  		S = sum(U);
			  		dst.at<float>(i,j) = s*S[0];
			  	}
		}
}

void fast_walsh_paley8(InputArray _src, OutputArray _dst){
	//
		Mat src = _src.getMat();
		float ncol = src.cols;
		float nrow = src.rows;

		_dst.create(nrow, ncol, CV_32F);
		Mat dst = _dst.getMat();

		Mat temp;
		Mat out = Mat(8, 8, CV_32F);
		Mat U;
		Scalar S;

		float had2[2][2] = { 1,  1,  1,  -1};
		Mat H2 = Mat(2, 2, CV_32F, had2);

		Mat I2 = Mat::eye(2, 2, CV_32F);
		Mat I4 = Mat::eye(4, 4, CV_32F);

		float t[1][2] = {1, 1};
		float b[1][2] = {1, -1};
		Mat top = Mat(1, 2, CV_32F, t);
		Mat bottom = Mat(1, 2, CV_32F, b);
		Mat p1t, p1b, p2t, p2b, P1, P2;

		kron_prod(I2, top, p1t);
		kron_prod(I2, bottom, p1b);
		kron_prod(I4, top, p2t);
		kron_prod(I4, bottom, p2b);

		vconcat(p1t, p1b, P1);
		vconcat(p2t, p2b, P2);

		Mat p0, p1, p2, WP4, WP8, WP8t;
		float s = (1/sqrt(8*8));

//8 point
		kron_prod(I2, H2, p0); //4x4
		WP4 = p0*P1;
		kron_prod(I4, H2, p1);
		kron_prod(I2, P1, p2);
		WP8 = p1*p2*P2;

		transpose(WP8, WP8t);

		for ( int i = 0; i < ncol; i++ ){
			  for (int j = 0; j < nrow; j++ ){
			  		temp = WP8t.col(i) * WP8.row(j);
			  		//transpose(temp, temp); //adjust for unequal matrices
			  		multiply(temp, src, U);
			  		S = sum(U);
			  		dst.at<float>(i,j) = s*S[0];
			  	}
		}
}

void ifast_walsh256(InputArray _src, OutputArray _dst ){
		Mat src = _src.getMat();
		float ncol = src.cols;
		float nrow = src.rows;

		_dst.create(nrow, ncol, CV_32F);
		Mat dst = _dst.getMat();

		Mat temp;
		Mat out = Mat(256, 256, CV_32F);
		Mat U;
		Scalar S;

		float had2[2][2] = { 1,  1,  1,  -1};
		Mat H2 = Mat(2, 2, CV_32F, had2);
		Mat H256 = Mat(256, 256, CV_32F);
		Mat H256t;

		Mat I2 = Mat::eye(2, 2, CV_32F);
		Mat I4 = Mat::eye(4, 4, CV_32F);
		Mat I8 = Mat::eye(8, 8, CV_32F);
		Mat I16 = Mat::eye(16, 16, CV_32F);
		Mat I32 = Mat::eye(32, 32, CV_32F);
		Mat I64 = Mat::eye(64, 64, CV_32F);
		Mat I128 = Mat::eye(128, 128, CV_32F);

		Mat B1, B2, B3, B4, B5, B6, B7, B8;
		Mat b;
		float s = (1/sqrt(256*256));

//256 point

		kron_prod(H2, I128, B1); //256x256

		kron_prod(H2, I64, b); //128x128
		kron_prod(I2, b, B2); //256x256

		kron_prod(H2, I32, b); //64x64
		kron_prod(I4, b, B3); //256x256

		kron_prod(H2, I16, b); //32x32
		kron_prod(I8, b, B4); //256x256

		kron_prod(H2, I8, b); //16x16
		kron_prod(I16, b, B5); //256x256

		kron_prod(H2, I4, b); //8x8
		kron_prod(I32, b, B6); //256x256

		kron_prod(H2, I2, b); //4x4
		kron_prod(I64, b, B7); //256x256

		kron_prod(I128, H2, B8); //256x256

		H256 = B1*B2*B3*B4*B5*B6*B7*B8;

		transpose(H256, H256t);

		for ( int i = 0; i < ncol; i++ ){
			  for (int j = 0; j < nrow; j++ ){
			  		temp = H256t.col(i) * H256.row(j);
			  		transpose(temp, temp); //adjust for unequal matrices
			  		multiply(temp, src, U);
			  		S = sum(U);
			  		dst.at<float>(i,j) = s*S[0];
			  	}
		}

//16 point
/*		Mat H16;
		kron_prod(H2, I8, B1); //16x16
		kron_prod(H2, I4, b0); //8x8
		kron_prod(I2, b0, B2); //16x16
		kron_prod(H2, I2, b0); //4x4
		kron_prod(I4, b0, B3); //16x16
		kron_prod(I8, H2, B4); //16x16

		H16 = B1*B2*B3*B4;
		std::cout << H16;
*/



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

#endif /* FAST_WALSH_HPP_ */

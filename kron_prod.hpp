/*
 * kron_prod.hpp
 *
 *  Created on: Feb 16, 2014
 *      Author: Carrie Hernandez
 *      Implement Kronecker product for 2 matrices
 */

#ifndef KRON_PROD_HPP_
#define KRON_PROD_HPP_

using namespace cv;
using namespace std;

void kron_prod( InputArray, InputArray, OutputArray);
/********************************************************************************/

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

#endif /* KRON_PROD_HPP_ */

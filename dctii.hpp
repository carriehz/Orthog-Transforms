/*
 * dctii.h
 *
 *  Created on: Feb 5, 2014
 *      Author: Carrie Hernandez
 *      Implementation of Discrete Cosine Transform II and Inverse DCT-II for m x n size matrix
 */

#ifndef DCTII_H_
#define DCTII_H_

//Generate matrix of DCT-IV coefficients
using namespace cv;
using namespace std;

void dctii( InputArray, OutputArray); // DCT-II of m x n size matrix
void idctii( InputArray, OutputArray); // Inverse DCT-II of m x n size matrix
/*******************************************************************************/
void dctii( InputArray _src, OutputArray _dst){

	Mat src = _src.getMat();
	float ncol = src.cols;
	float nrow = src.rows;

	_dst.create(nrow, ncol, CV_32F);
	Mat dst = _dst.getMat();
	Mat c1(1, ncol, CV_32F);
	Mat c2(1, nrow, CV_32F);
	Mat um(ncol, ncol, CV_32F);
	Mat umt(ncol, ncol, CV_32F);
	Mat un(nrow, nrow, CV_32F);
	Mat U;
	Mat temp;
	Scalar S;

	  	 for( int i = 0; i < ncol; i++ ){
	  		 c1.at<float>(i)=sqrt(2.0f/ncol);
	  	 }
	  	 c1.at<float>(0,0)=sqrt(2.0f/ncol)/sqrt(2.0f);

	  	 for( int k = 0; k < ncol; k++ ){
	  		 for( int n = 0; n < ncol; n++ ){

	  			um.at<float>(k,n)=c1.at<float>(k)*cos( (CV_PI*(k))*((2.0f*n)+1)/(2.0f*ncol));
	  		 }
	  	 }
	  	 for( int i = 0; i < nrow; i++ ){
	  	  		 c2.at<float>(i)=sqrt(2.0f/nrow);
	  	  	 }
	  	 c2.at<float>(0,0)=sqrt(2.0f/nrow)/sqrt(2.0f);

	  	for( int k = 0; k < nrow; k++ ){
	  	  		 for( int n = 0; n < nrow; n++ ){

	  	  			un.at<float>(k,n)=c2.at<float>(k)*cos( (CV_PI*(k))*((2.0f*n)+1)/(2*nrow));
	  	  	  	}
	  	  	 }

	  	transpose(um, umt);

	  	dst = (um*src)*umt;

/*		  ofstream myfilea ("dct_result.txt");
		  		if (myfilea.is_open()){
		  			myfilea << dst;
		  			myfilea.close();
		  			}
		  		else cout << "Unable to open file";

	imwrite("/Users/me/cuda-workspace/Project3/dct_result.bmp", dst);
*/
}
void idctii( InputArray _src, OutputArray _dst){

	Mat src = _src.getMat();
	float ncol = src.cols;
	float nrow = src.rows;

	_dst.create(nrow, ncol, CV_32F);
	Mat dst = _dst.getMat();
	Mat c1(1, ncol, CV_32F);
	Mat c2(1, nrow, CV_32F);
	Mat um(ncol, ncol, CV_32F);
	Mat umt(ncol, ncol, CV_32F);
	Mat un(nrow, nrow, CV_32F);
	Mat U;
	Mat temp;
	Scalar S;

	  	 for( int i = 0; i < ncol; i++ ){
	  		 c1.at<float>(i)=sqrt(2.0f/ncol);
	  	 }
	  	 c1.at<float>(0,0)=sqrt(2.0f/ncol)/sqrt(2.0f);

	  	 for( int k = 0; k < ncol; k++ ){
	  		 for( int n = 0; n < ncol; n++ ){

	  			um.at<float>(k,n)=c1.at<float>(k)*cos( (CV_PI*(k))*((2.0f*n)+1)/(2.0f*ncol));
	  		 }
	  	 }
	  	 for( int i = 0; i < nrow; i++ ){
	  	  		 c2.at<float>(i)=sqrt(2.0f/nrow);
	  	  	 }
	  	 c2.at<float>(0,0)=sqrt(2.0f/nrow)/sqrt(2.0f);

	  	for( int k = 0; k < nrow; k++ ){
	  	  		 for( int n = 0; n < nrow; n++ ){

	  	  			un.at<float>(k,n)=c2.at<float>(k)*cos( (CV_PI*(k))*((2.0f*n)+1)/(2*nrow));
	  	  	  	}
	  	  	 }

	  	transpose(um, umt);

	  	dst = (umt*src)*um;



}

#endif /* DCTII_H_ */

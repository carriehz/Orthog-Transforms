/*
 * para_slant.hpp
 *  Created on: Feb 16, 2014
 *      Author: Carrie Hernandez
* Implementation of a Parametric Slant Transform as defined in ...
* Agaian, S., Tourshan, K., Noonan, J.  "Parametric Slant-Hadamard Transforms With Applications," ...
* IEEE Signal Processing Letters, Vol 9, No 11, November 2002.
*
* Parametric Slant-Hadamard Tranforms and Inverse for 256 x 256 and 8 x 8 pixel images
* Single or Multiple Betas
* Kronecker Product function
 */

#ifndef PARA_SLANT_HPP_
#define PARA_SLANT_HPP_

using namespace cv;
using namespace std;

/**********************************************************************/

void para_slant256(InputArray, OutputArray);
void para_slant8(InputArray, OutputArray);
void ipara_slant256(InputArray, OutputArray);
void ipara_slant8(InputArray, OutputArray);
//void kron_prod( InputArray, InputArray, OutputArray);


/*******************************************************************/
void para_slant256(InputArray _src, OutputArray _dst){

		Mat src = _src.getMat();
		float ncol = src.cols;
		float nrow = src.rows;
		_dst.create(nrow, ncol, CV_32F);
		Mat dst = _dst.getMat();
		Mat temp, U;
		Mat out = Mat(256, 256, CV_32F);
		Scalar S;

		float n = 8; // 256 x 256 size (2^8)
//constant betas
		//float beta = 1; // constant beta for classic slant
		float beta = 1.7; // constant beta

		float a2n, b2n, at, bt, p2n;
		Mat Op, Ip;
		Mat R, R0, R1, R2, R3, R4, R5, R6;
		Mat Q1, Q2, Q3, Q4, Q, q, q0, q1, q2, q3, q4, qr, qc;
		Mat S2, s4, S4, S8, S8t;
		Mat s8;
		Mat T, Tt;
		float s;
		float had2[2][2] = { 1,  1,  1,  -1};
		Mat H2 = Mat(2, 2, CV_32F, had2);
		Mat H4;
		Mat I2 = Mat::eye(2, 2, CV_32F);
		kron_prod(H2, H2, H4);

	for( int i = 2; i < n+1; i++ ){

	//calculate a2^n and b2^n
			float r = 2.0f*(i-1); 		//power of 2 for iteration
			p2n = pow(2, r);		//2^2*(n-1)
			at = ((3.0f*p2n)/((4.0f*p2n)-beta));
			bt = ((p2n - beta)/((4.0f*p2n) - beta));
			a2n = sqrt(at);
			b2n = sqrt(bt);

			float m1[2][2] = {1, 0, 0, b2n};
			Mat M1 = Mat(2, 2, CV_32F, m1);
			float m2[2][2] = {0, 0, a2n, 0};
			Mat M2 = Mat(2, 2, CV_32F, m2);
			float m3[2][2] = {0, 0, 0, a2n};
			Mat M3 = Mat(2, 2, CV_32F, m3);
			float m4[2][2] = {0, 1, -b2n, 0};
			Mat M4 = Mat(2, 2, CV_32F, m4);

		if (i == 2){
	//M2n matrix
				hconcat(M1, M2, S2);
				hconcat(M3, M4, S4);
				vconcat(S2, S4, Q);
				s4 = Q*H4;
				kron_prod(I2, s4, s4);
				s4.copyTo(T);
		}

		else{
//Zero and Identity Matrices
			float r2 = (i-1);
			float p2 = pow(2,r2) - 2;   // 2^(n-1) - 2
			Mat O = Mat::zeros(p2, p2, CV_32F);
			Mat I = Mat::eye(p2, p2, CV_32F);
			Mat Or = Mat::zeros(1, p2, CV_32F);
			Mat Oc = Mat::zeros(p2, 1, CV_32F);
		//Q2n matrix
			Mat K, Kk, nK;
			vconcat(Or, Or, qr);
			hconcat(Oc, Oc, qc);
			hconcat(qc, I, q1);
			hconcat(qc, O, q0);
			hconcat(M1, qr, R1);
			hconcat(M2, qr, R2);
			hconcat(M3, qr, R3);
			hconcat(M4, qr, R4);
			hconcat(R1, R2, R0);
			hconcat(R3, R4, R);
			hconcat(q1, q0, q);
			vconcat(R0, q, q2);
			hconcat(q0, q1, q3);
			vconcat(R, q3, q4);
			vconcat(q2, q4, Q);

			if (i == 3){
				s8 = Q*T;
				s8.copyTo(T);
			}
			else {

				kron_prod(I2, T, T);
				s8 = Q*T;
				s8.copyTo(T);
			}
		}
	}
	s = (1/sqrt(4));
	multiply(s, T, T);
		transpose(T, Tt);
		out = (T*src)*Tt;
		out.copyTo(dst);
}

void para_slant8(InputArray _src, OutputArray _dst){
	//
		Mat src = _src.getMat();
		float ncol = src.cols;
		float nrow = src.rows;
		_dst.create(nrow, ncol, CV_32F);
		Mat dst = _dst.getMat();
		Mat temp, U;
		Mat out = Mat(8, 8, CV_32F);
		Scalar S;

		float n = 3;
//constant betas
		float beta = 1;  // classic slant transform
		//float beta = 1.7;
		float a2n, b2n, at, bt, p2n;
		Mat Op, Ip;
		Mat R1, R2, R3, R4;
		Mat Q1, Q2, Q3, Q4, Q8, q,q0, q1, q2, q3, q4;
		Mat S2, s4, S4, S8, S8t;
		Mat s8;

		float had2[2][2] = { 1,  1,  1,  -1};
		Mat H2 = Mat(2, 2, CV_32F, had2);
		Mat s2 = H2;
		Mat I2 = Mat::eye(2, 2, CV_32F);
		kron_prod(I2, s2, S2);

	for( int i = 2; i < n+1; i++ ){
		//zero matrix and identity matrix power of 2
		if (i == 2){

	//Multiple betas
			//float beta = 4.0; // beta of 4 and 16 equals hadamard transform
	//calculate a2^n and b2^n
			float r = 2.0f*(i-1); 				//power of 2 for iteration
			p2n = pow(2.0f, r);			 //2^2*(n-1)
			at = ((3.0f*p2n)/((4.0f*p2n)-beta));
			bt = ((p2n - beta)/((4.0f*p2n) - beta));
			a2n = sqrt(at);
			b2n = sqrt(bt);
			float m1[2][2] = {1, 0, a2n, b2n};
			Mat M1 = Mat(2, 2, CV_32F, m1);
			float m2[2][2] = {1, 0, -a2n, b2n};
			Mat M2 = Mat(2, 2, CV_32F, m2);
			float m3[2][2] = {0, 1, -b2n, a2n};
			Mat M3 = Mat(2, 2, CV_32F, m3);
			float m4[2][2] = {0, -1, b2n, a2n};
			Mat M4 = Mat(2, 2, CV_32F, m4);

			hconcat(M1, M2, q1);
			hconcat(M3, M4, q2);
			vconcat(q1, q2, Q4);
			s4 = Q4*S2;
			kron_prod(I2, s4, s4);
			s4.copyTo(S4);
		}
		else{

//Multiple betas
			//float beta = 16.0;     // beta of 4 and 16 equals hadamard transform
//a2n and b2n values
			float r = (float)2*(i-1); 				//power of 2 for iteration
			p2n = pow(2, r);			 //2^2*(n-1)
			at = ((3.0f*p2n)/((4.0f*p2n)-beta));
			bt = ((p2n - beta)/((4.0f*p2n) - beta));
			a2n = sqrt(at);
			b2n = sqrt(bt);

			float m1[2][2] = {1, 0, a2n, b2n};
			Mat M1 = Mat(2, 2, CV_32F, m1);
			float m2[2][2] = {1, 0, -a2n, b2n};
			Mat M2 = Mat(2, 2, CV_32F, m2);
			float m3[2][2] = {0, 1, -b2n, a2n};
			Mat M3 = Mat(2, 2, CV_32F, m3);
			float m4[2][2] = {0, -1, b2n, a2n};
			Mat M4 = Mat(2, 2, CV_32F, m4);

//Zero and Identity Matrices
			float r2 = (i-1);
			float p2 = pow(2,r2) - 2;   // 2^(n-1) - 2
			Mat O = Mat::zeros(p2, p2, CV_32F);
			Mat I = Mat::eye(p2, p2, CV_32F);
			O.copyTo(Op);
			I.copyTo(Ip);
		//Q2n matrix
			Mat K;
			hconcat(Op, Ip, K);
			hconcat(M1, Op, R1);
			vconcat(R1, K, q1);
			hconcat(M2, Op, R2);
			vconcat(R2, K, q2);
			hconcat(q1, q2, q0);
			hconcat(M3, Op, R3);
			vconcat(R3, K, q3);
			hconcat(M4, Op, R4);
			vconcat(R4, -K, q4);
			hconcat(q3, q4, q);
			vconcat(q0, q, Q8);

			s8 = Q8*S4;
			float s = 1/(sqrt(8.0f));
			multiply(s, s8, s8 );
			s8.copyTo(S8);
			}
	}

//8 point

		transpose(S8, S8t);
		dst = (S8*src)*S8t;
}

void ipara_slant256(InputArray _src, OutputArray _dst){
	//
		Mat src = _src.getMat();
		float ncol = src.cols;
		float nrow = src.rows;
		_dst.create(nrow, ncol, CV_32F);
		Mat dst = _dst.getMat();

		Mat temp, U;
		Mat out = Mat(256, 256, CV_32F);
		Scalar S;

		float n = 8;
//constant betas
		float beta = 1;
		//float beta = 1.7;

		float a2n, b2n, at, bt, p2n;
		Mat Op, Ip;
		Mat R, R0, R1, R2, R3, R4, R5, R6;
		Mat Q1, Q2, Q3, Q4, Q, q, q0, q1, q2, q3, q4, qr, qc;
		Mat S2, s4, S4, S8, S8t;
		Mat s8;
		Mat T;
		Mat Tt;
		float s;

		float had2[2][2] = { 1,  1,  1,  -1};
		Mat H2 = Mat(2, 2, CV_32F, had2);
		Mat H4;

		Mat I2 = Mat::eye(2, 2, CV_32F);
		kron_prod(H2, H2, H4);

	for( int i = 2; i < n+1; i++ ){

	//calculate a2^n and b2^n
			float r = 2.0f*(i-1); 				//power of 2 for iteration
			p2n = pow(2, r);			 //2^2*(n-1)
			at = ((3.0f*p2n)/((4.0f*p2n)-beta));
			bt = ((p2n - beta)/((4.0f*p2n) - beta));
			a2n = sqrt(at);
			b2n = sqrt(bt);

			float m1[2][2] = {1, 0, 0, b2n};
			Mat M1 = Mat(2, 2, CV_32F, m1);
			float m2[2][2] = {0, 0, a2n, 0};
			Mat M2 = Mat(2, 2, CV_32F, m2);
			float m3[2][2] = {0, 0, 0, a2n};
			Mat M3 = Mat(2, 2, CV_32F, m3);
			float m4[2][2] = {0, 1, -b2n, 0};
			Mat M4 = Mat(2, 2, CV_32F, m4);

		if (i == 2){

	//M2n matrix
				hconcat(M1, M2, S2);
				hconcat(M3, M4, S4);
				vconcat(S2, S4, Q);
				s4 = Q*H4;
				kron_prod(I2, s4, s4);
				s4.copyTo(T);
		}

		else{
//Zero and Identity Matrices
			float r2 = (i-1);
			float p2 = pow(2,r2) - 2;   // 2^(n-1) - 2
			Mat O = Mat::zeros(p2, p2, CV_32F);
			Mat I = Mat::eye(p2, p2, CV_32F);
			Mat Or = Mat::zeros(1, p2, CV_32F);
			Mat Oc = Mat::zeros(p2, 1, CV_32F);
		//Q2n matrix
			Mat K, Kk, nK;
			vconcat(Or, Or, qr);
			hconcat(Oc, Oc, qc);
			hconcat(qc, I, q1);
			hconcat(qc, O, q0);
			hconcat(M1, qr, R1);
			hconcat(M2, qr, R2);
			hconcat(M3, qr, R3);
			hconcat(M4, qr, R4);
			hconcat(R1, R2, R0);
			hconcat(R3, R4, R);
			hconcat(q1, q0, q);
			vconcat(R0, q, q2);
			hconcat(q0, q1, q3);
			vconcat(R, q3, q4);
			vconcat(q2, q4, Q);

			if (i == 3){
				s8 = Q*T;
				s8.copyTo(T);

			}
			else {

				kron_prod(I2, T, T);
				s8 = Q*T;
				s8.copyTo(T);
			}
		}

	}

	s = (1/sqrt(4));
	multiply(s, T, T);
//
		transpose(T, Tt);
		dst = (Tt*src)*T;
}

void ipara_slant8(InputArray _src, OutputArray _dst){
	//
	Mat src = _src.getMat();
	float ncol = src.cols;
	float nrow = src.rows;
	_dst.create(nrow, ncol, CV_32F);
	Mat dst = _dst.getMat();

	Mat temp, U;
	Mat out = Mat(8, 8, CV_32F);
	Scalar S;

	float n = 3;

//constant betas
	//float beta = 1;
	float beta = 1.7;

	float a2n, b2n, at, bt, p2n;
	Mat Op, Ip;
	Mat R1, R2, R3, R4;
	Mat Q1, Q2, Q3, Q4, Q8, q,q0, q1, q2, q3, q4;
	Mat S2, s4, S4, S8, S8t;
	Mat s8;

	float had2[2][2] = { 1,  1,  1,  -1};
	Mat H2 = Mat(2, 2, CV_32F, had2);

	//Mat s2 = (float)(1/sqrt(2))*H2;
	Mat s2 = H2;
	Mat I2 = Mat::eye(2, 2, CV_32F);
	kron_prod(I2, s2, S2);

for( int i = 2; i < n+1; i++ ){
	//zero matrix and identity matrix power of 2

	if (i == 2){
//Multiple betas
		//float beta = 4.0;
//calculate a2^n and b2^n
		float r = 2.0f*(i-1); 				//power of 2 for iteration
		p2n = pow(2.0f, r);			 //2^2*(n-1)
		at = ((3.0f*p2n)/((4.0f*p2n)-beta));
		bt = ((p2n - beta)/((4.0f*p2n) - beta));
		a2n = sqrt(at);
		b2n = sqrt(bt);
		float m1[2][2] = {1, 0, a2n, b2n};
		Mat M1 = Mat(2, 2, CV_32F, m1);
		float m2[2][2] = {1, 0, -a2n, b2n};
		Mat M2 = Mat(2, 2, CV_32F, m2);
		float m3[2][2] = {0, 1, -b2n, a2n};
		Mat M3 = Mat(2, 2, CV_32F, m3);
		float m4[2][2] = {0, -1, b2n, a2n};
		Mat M4 = Mat(2, 2, CV_32F, m4);

		hconcat(M1, M2, q1);
		hconcat(M3, M4, q2);
		vconcat(q1, q2, Q4);
		s4 = Q4*S2;
		kron_prod(I2, s4, s4);
		s4.copyTo(S4);
	}
	else{

//Multiple betas
		//float beta = 16.0;

//a2n and b2n values
		float r = (float)2*(i-1); 				//power of 2 for iteration
		p2n = pow(2, r);			 //2^2*(n-1)
		at = ((3.0f*p2n)/((4.0f*p2n)-beta));
		bt = ((p2n - beta)/((4.0f*p2n) - beta));
		a2n = sqrt(at);
		b2n = sqrt(bt);

		float m1[2][2] = {1, 0, a2n, b2n};
		Mat M1 = Mat(2, 2, CV_32F, m1);
		float m2[2][2] = {1, 0, -a2n, b2n};
		Mat M2 = Mat(2, 2, CV_32F, m2);
		float m3[2][2] = {0, 1, -b2n, a2n};
		Mat M3 = Mat(2, 2, CV_32F, m3);
		float m4[2][2] = {0, -1, b2n, a2n};
		Mat M4 = Mat(2, 2, CV_32F, m4);

//Zero and Identity Matrices
		float r2 = (i-1);
		float p2 = pow(2,r2) - 2;   // 2^(n-1) - 2
		Mat O = Mat::zeros(p2, p2, CV_32F);
		Mat I = Mat::eye(p2, p2, CV_32F);
		O.copyTo(Op);
		I.copyTo(Ip);

	//Q2n matrix

		Mat K;
		hconcat(Op, Ip, K);
		hconcat(M1, Op, R1);
		vconcat(R1, K, q1);
		hconcat(M2, Op, R2);
		vconcat(R2, K, q2);
		hconcat(q1, q2, q0);
		hconcat(M3, Op, R3);
		vconcat(R3, K, q3);
		hconcat(M4, Op, R4);
		vconcat(R4, -K, q4);
		hconcat(q3, q4, q);
		vconcat(q0, q, Q8);

		s8 = Q8*S4;

		float s = 1/(sqrt(8.0f));
		multiply(s, s8, s8 );
		s8.copyTo(S8);
		}
}

		transpose(S8, S8t);
		dst = (S8t*src)*S8;
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
#endif /* PARA_SLANT_HPP_ */

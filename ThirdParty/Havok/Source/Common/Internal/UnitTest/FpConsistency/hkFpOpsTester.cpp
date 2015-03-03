/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Internal/hkInternal.h>

#include "hkFpOpsTester.h"

#include <Common/Base/Container/String/hkStringBuf.h>

#if defined HK_COMPILER_MSVC
//div instruction may be inaccurate on some pentiums,
//but that is the goal of the test
#pragma warning(disable: 4725)
#endif


#define ENABLE_FP_TEST
#define ENABLE_SIMD_TEST
#define ENABLE_DIV_TEST
#define ENABLE_CONTRACTION_TEST


float _cancelOpFp(float arg1, float arg2, float arg3, float arg4)
{
	float intermediate = arg1 + arg2;
	intermediate *= arg3;
	intermediate -= arg4;

	return intermediate;
}

float _cancelOp2Fp(float arg1, float arg2)
{
	float intermediate = arg1 + arg2;
	intermediate -= arg1;

	return intermediate;
}

float _testAddFp(float arg1, float arg2)
{
	float intermediate = arg1 + arg2;

	return intermediate;
}

float _testSubFp(float arg1, float arg2)
{
	float intermediate = arg1 - arg2;

	return intermediate;
}

float _testMulFp(float arg1, float arg2)
{
	float intermediate = arg1 * arg2;

	return intermediate;
}

float _testDivFp(float arg1, float arg2)
{
	float intermediate = arg1 / arg2;

	return intermediate;
}

float _testSqrtFp(float arg1)
{
	float intermediate = hkMath::sqrt(arg1);

	return intermediate;
}

float _testSinFp(float arg1)
{
	float intermediate = hkMath::sin(arg1);

	return intermediate;
}

float _testCosFp(float arg1)
{
	float intermediate = hkMath::cos(arg1);

	return intermediate;
}

float _testTanFp(float arg1)
{
	float intermediate = tanf(arg1);

	return intermediate;
}

float _testLogFp(float arg1)
{
	float intermediate = hkMath::log(arg1);

	return intermediate;
}

float _testPowFp(float arg1)
{
	float intermediate = hkMath::pow(arg1,arg1);

	return intermediate;
}

float _testAcosFp(float arg1)
{
	float intermediate = hkMath::acos(arg1);

	return intermediate;
}

float _testAsinFp(float arg1)
{
	float intermediate = hkMath::asin(arg1);

	return intermediate;
}

float _testAtan2Fp(float arg1, float arg2)
{
	float intermediate = hkMath::atan2(arg1,arg2);

	return intermediate;
}

float _testContractionFp(float arg1, float arg2, float arg3)
{
	float intermediate = (arg1 * arg2) + arg3;

	return intermediate;
}

double _testAddDouble(double arg1, double arg2)
{
	double intermediate = arg1 + arg2;

	return intermediate;
}

double _testMulDouble(double arg1, double arg2)
{
	double intermediate = arg1 * arg2;

	return intermediate;
}

double _testDivDouble(double arg1, double arg2)
{
	double intermediate = arg1 / arg2;

	return intermediate;
}


#ifdef ENABLE_SIMD_TEST
float _cancelOpSimd(float arg1, float arg2, float arg3, float arg4)
{
	hkSimdReal arg1Simd; arg1Simd.setFromFloat(arg1);
	hkSimdReal arg2Simd; arg2Simd.setFromFloat(arg2);
	hkSimdReal arg3Simd; arg3Simd.setFromFloat(arg3);
	hkSimdReal arg4Simd; arg4Simd.setFromFloat(arg4);
	hkSimdReal resultSimd;

	hkSimdReal intermediate = arg1Simd + arg2Simd;
	intermediate = intermediate * arg3Simd;
	intermediate = intermediate - arg4Simd;

	resultSimd = intermediate;
	float result = (float)resultSimd.getReal();
	return result;
}

float _cancelOp2Simd(float arg1, float arg2)
{
	hkSimdReal arg1Simd; arg1Simd.setFromFloat(arg1);
	hkSimdReal arg2Simd; arg2Simd.setFromFloat(arg2);
	hkSimdReal resultSimd;

	hkSimdReal intermediate = arg1Simd + arg2Simd;
	intermediate.sub(arg1Simd);

	float result = (float)resultSimd.getReal();
	return result;
}

float _testAddSimd(float arg1, float arg2)
{
	hkSimdReal arg1Simd; arg1Simd.setFromFloat(arg1);
	hkSimdReal arg2Simd; arg2Simd.setFromFloat(arg2);
	hkSimdReal resultSimd;

	resultSimd = arg1Simd + arg2Simd;

	float result = (float)resultSimd.getReal();
	return result;
}

float _testSubSimd(float arg1, float arg2)
{
	hkSimdReal arg1Simd; arg1Simd.setFromFloat(arg1);
	hkSimdReal arg2Simd; arg2Simd.setFromFloat(arg2);
	hkSimdReal resultSimd;

	resultSimd = arg1Simd - arg2Simd;

	float result = (float)resultSimd.getReal();
	return result;
}

float _testMulSimd(float arg1, float arg2)
{
	hkSimdReal arg1Simd; arg1Simd.setFromFloat(arg1);
	hkSimdReal arg2Simd; arg2Simd.setFromFloat(arg2);
	hkSimdReal resultSimd;

	resultSimd = arg1Simd * arg2Simd;

	float result = (float)resultSimd.getReal();
	return result;
}

float _testDivSimd(float arg1, float arg2)
{
	hkSimdReal arg1Simd; arg1Simd.setFromFloat(arg1);
	hkSimdReal arg2Simd; arg2Simd.setFromFloat(arg2);
	hkSimdReal resultSimd;

	resultSimd = arg1Simd / arg2Simd;

	float result = (float)resultSimd.getReal();
	return result;
}

float _testSqrtSimd(float arg1)
{
	hkSimdReal arg1Simd; arg1Simd.setFromFloat(arg1);
	hkSimdReal resultSimd;

	resultSimd = arg1Simd.sqrt();

	float result = (float)resultSimd.getReal();
	return result;
}

float _testContractionSimd(float arg1, float arg2, float arg3)
{
	hkSimdReal arg1Simd; arg1Simd.setFromFloat(arg1);
	hkSimdReal arg2Simd; arg2Simd.setFromFloat(arg2);
	hkSimdReal arg3Simd; arg3Simd.setFromFloat(arg3);
	hkSimdReal resultSimd;

	resultSimd = (arg1Simd * arg2Simd) + arg3Simd;

	float result = (float)resultSimd.getReal();
	return result;
}
#endif

void flushNaN(float& argument)
{
	unsigned int argumentInt = *((int*)(&argument));
	unsigned int argExp = (argumentInt & 0x7f800000);
	if( argExp == 0x7f800000 )
	{
		argumentInt = ((argumentInt & 0x007fffff) == 0) ? argumentInt : (argExp | 0x1);
		argument = *((float*)(&argumentInt));
	}
}

void flushNaNDenorm(float& argument)
{
	unsigned int argumentInt = *((int*)(&argument));
	unsigned int argExp = (argumentInt & 0x7f800000);
	if( argExp == 0x7f800000 )
	{
		argumentInt = ((argumentInt & 0x007fffff) == 0) ? argumentInt : (argExp | 0x1);
		argument = *((float*)(&argumentInt));
	}
	else if( argExp == 0x0 || argExp == 0x00800000 )
	{
		argumentInt = (argumentInt & 0x00000000);
		argument = *((float*)(&argumentInt));
	}
}

void flushSpecials(float& argument)
{
	//flushNaN(argument);
}

void hkFpOpsTester::cancelOpFp(float arg1, float arg2, float arg3, float arg4)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _cancelOpFp(arg1,arg2,arg3,arg4);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
#endif
}

void hkFpOpsTester::cancelOp2Fp(float arg1, float arg2)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _cancelOp2Fp(arg1,arg2);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
#endif
}

float hkFpOpsTester::testAddFp(float arg1, float arg2)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _testAddFp(arg1,arg2);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));

	return fpResult;
#else
	return 0.0f;
#endif
}

float hkFpOpsTester::testSubFp(float arg1, float arg2)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _testSubFp(arg1,arg2);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
	return fpResult;
#else
	return 0.0f;
#endif
}

float hkFpOpsTester::testMulFp(float arg1, float arg2)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _testMulFp(arg1,arg2);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
	return fpResult;
#else
	return 0.0f;
#endif
}

float hkFpOpsTester::testDivFp(float arg1, float arg2)
{
#ifdef ENABLE_FP_TEST
#ifdef ENABLE_DIV_TEST
	float fpResult = _testDivFp(arg1,arg2);
	flushSpecials(fpResult);
	m_crcWriterFpDiv.write(&fpResult,sizeof(float));
	return fpResult;
#endif
#endif
}

float hkFpOpsTester::testSqrtFp(float arg1)
{
#ifdef ENABLE_FP_TEST
	float fpResult;
	{
		fpResult = _testSqrtFp(arg1);
		flushSpecials(fpResult);
		m_crcWriterFp.write(&fpResult,sizeof(float));
	}
	return fpResult;
#endif
}

float hkFpOpsTester::testSinFp(float arg1)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _testSinFp(arg1);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
	return fpResult;
#endif
}

float hkFpOpsTester::testCosFp(float arg1)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _testCosFp(arg1);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
	return fpResult;
#endif
}

float hkFpOpsTester::testTanFp(float arg1)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _testTanFp(arg1);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
	return fpResult;
#endif
}

float hkFpOpsTester::testLogFp(float arg1)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _testLogFp(arg1);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
	return fpResult;
#endif
}

float hkFpOpsTester::testAtan2Fp(float arg1, float arg2)
{
#ifdef ENABLE_FP_TEST
	float fpResult = _testAtan2Fp(arg1, arg2);
	flushSpecials(fpResult);
	m_crcWriterFp.write(&fpResult,sizeof(float));
	return fpResult;
#endif
}

void hkFpOpsTester::testContractionFp(float arg1, float arg2, float arg3)
{
#ifdef ENABLE_FP_TEST
#ifdef ENABLE_CONTRACTION_TEST
	float fpResult = _testContractionFp(arg1, arg2, arg3);
	m_crcWriterFpContract.write(&fpResult,sizeof(float));
#endif
#endif
}

double hkFpOpsTester::testAddDouble(double arg1, double arg2)
{
#ifdef ENABLE_FP_TEST
	hkFpOpsTester::DoubleUnion doubleResult;
	doubleResult.m_float64 = _testAddDouble(arg1, arg2);
	m_crcWriterFp.write(&(doubleResult.m_float64),sizeof(double));
	//hkStringBuf sb; sb.printf( "Double test result: 0x%x%x \n", doubleResult.m_int32[1], doubleResult.m_int32[0] );

	return doubleResult.m_float64;
#endif
}

double hkFpOpsTester::testMulDouble(double arg1, double arg2)
{
#ifdef ENABLE_FP_TEST
	hkFpOpsTester::DoubleUnion doubleResult;
	doubleResult.m_float64 = _testMulDouble(arg1, arg2);
	m_crcWriterFp.write(&(doubleResult.m_float64),sizeof(double));
	//hkStringBuf sb; sb.printf( "Double test result: 0x%x%x \n", doubleResult.m_int32[1], doubleResult.m_int32[0] );

	return doubleResult.m_float64;
#endif
}

double hkFpOpsTester::testDivDouble(double arg1, double arg2)
{
#ifdef ENABLE_FP_TEST
#ifdef ENABLE_DIV_TEST
	hkFpOpsTester::DoubleUnion doubleResult;
	doubleResult.m_float64 = _testDivDouble(arg1, arg2);
	m_crcWriterFpDiv.write(&(doubleResult.m_float64),sizeof(double));

	return doubleResult.m_float64;
#endif
#endif
}

double hkFpOpsTester::testAddDoubleSimd(double arg1, double arg2)
{
#ifdef ENABLE_SIMD_TEST
	hkFpOpsTester::DoubleUnion doubleResult;

#ifndef _WIN64
	__asm
	{
		movsd xmm0, arg1;
		movsd xmm1, arg2;
		addsd xmm0, xmm1;
		movsd doubleResult.m_float64, xmm0;
	}
#else
	doubleResult.m_float64 = 0.0;
#endif

	m_crcWriterSimd.write(&(doubleResult.m_float64),sizeof(double));

	return doubleResult.m_float64;
#endif
}
double hkFpOpsTester::testMulDoubleSimd(double arg1, double arg2)
{
#ifdef ENABLE_SIMD_TEST
	hkFpOpsTester::DoubleUnion doubleResult;

#ifndef _WIN64
	__asm
	{
		movsd xmm0, arg1;
		movsd xmm1, arg2;
		mulsd xmm0, xmm1;
		movsd doubleResult.m_float64, xmm0;
	}
#else
	doubleResult.m_float64 = 0.0;
#endif

	m_crcWriterSimd.write(&(doubleResult.m_float64),sizeof(double));

	return doubleResult.m_float64;
#endif
}
double hkFpOpsTester::testDivDoubleSimd(double arg1, double arg2)
{
#ifdef ENABLE_SIMD_TEST
#ifdef ENABLE_DIV_TEST
	hkFpOpsTester::DoubleUnion doubleResult;

#ifndef _WIN64
	__asm
	{
		movsd xmm0, arg1;
		movsd xmm1, arg2;
		divsd xmm0, xmm1;
		movsd doubleResult.m_float64, xmm0;
	}
#else
	doubleResult.m_float64 = 0.0;
#endif

	m_crcWriterSimdDiv.write(&(doubleResult.m_float64),sizeof(double));

	return doubleResult.m_float64;
#endif
#endif
}

void hkFpOpsTester::cancelOpSimd(float arg1, float arg2, float arg3, float arg4)
{
#ifdef ENABLE_SIMD_TEST
	float simdResult = _cancelOpSimd(arg1,arg2,arg3,arg4);
	flushSpecials(simdResult);
	m_crcWriterSimd.write(&simdResult,sizeof(float));
#endif
}

void hkFpOpsTester::cancelOp2Simd(float arg1, float arg2)
{
#ifdef ENABLE_SIMD_TEST
	float simdResult = _cancelOp2Simd(arg1,arg2);
	flushSpecials(simdResult);
	m_crcWriterSimd.write(&simdResult,sizeof(float));
#endif
}

float hkFpOpsTester::testAddSimd(float arg1, float arg2)
{
#ifdef ENABLE_SIMD_TEST
	float simdResult = _testAddSimd(arg1,arg2);
	flushSpecials(simdResult);
	m_crcWriterSimd.write(&simdResult,sizeof(float));

	return simdResult;
#else
	return 0.0f;
#endif
}

float hkFpOpsTester::testSubSimd(float arg1, float arg2)
{
#ifdef ENABLE_SIMD_TEST
	float simdResult = _testSubSimd(arg1,arg2);
	flushSpecials(simdResult);
	m_crcWriterSimd.write(&simdResult,sizeof(float));
	return simdResult;
#else
	return 0.0f;
#endif
}

float hkFpOpsTester::testMulSimd(float arg1, float arg2)
{
#ifdef ENABLE_SIMD_TEST
	float simdResult = _testMulSimd(arg1,arg2);
	flushSpecials(simdResult);
	m_crcWriterSimd.write(&simdResult,sizeof(float));
	return simdResult;
#else
	return 0.0f;
#endif
}

float hkFpOpsTester::testDivSimd(float arg1, float arg2)
{
#ifdef ENABLE_SIMD_TEST
#ifdef ENABLE_DIV_TEST
	float simdResult = _testDivSimd(arg1,arg2);
	flushSpecials(simdResult);
	m_crcWriterSimdDiv.write(&simdResult,sizeof(float));
	return simdResult;
#endif
#endif
}

float hkFpOpsTester::testSqrtSimd(float arg1)
{
#ifdef ENABLE_SIMD_TEST
#ifdef ENABLE_DIV_TEST
	float simdResult = _testSqrtSimd(arg1);
	flushSpecials(simdResult);
	m_crcWriterSimd.write(&simdResult,sizeof(float));
	return simdResult;
#endif
#endif
}

void hkFpOpsTester::testContractionSimd(float arg1, float arg2, float arg3)
{
#ifdef ENABLE_SIMD_TEST
#ifdef ENABLE_CONTRACTION_TEST
	float simdResult = _testContractionSimd(arg1, arg2, arg3);
	m_crcWriterSimdContract.write(&simdResult,sizeof(float));
#endif
#endif
}

unsigned int hkFpOpsTester::getCrcFp()
{ 
	return m_crcWriterFp.getCrc(); 
}

unsigned int hkFpOpsTester::getCrcFpDiv()
{ 
	return m_crcWriterFpDiv.getCrc(); 
}

unsigned int hkFpOpsTester::getCrcFpContract()
{ 
	return m_crcWriterFpContract.getCrc(); 
}

unsigned int hkFpOpsTester::getCrcSimd()
{ 
	return m_crcWriterSimd.getCrc(); 
}

unsigned int hkFpOpsTester::getCrcSimdDiv()
{ 
	return m_crcWriterSimdDiv.getCrc(); 
}

unsigned int hkFpOpsTester::getCrcSimdContract()
{ 
	return m_crcWriterSimdContract.getCrc(); 
}


float hkFpOpsTester::unregistered_cancelOp()
{
	int oneConstant = 0x3f800000;
	int smallConstant = 0x33800000;

	//int denormConstant1 = 0x00004000;
	//int denormConstant2 = 0x00000001;
	
	float test1 = *((float*)(&oneConstant));
	float test2 = *((float*)(&smallConstant));
	//float test1 = *((float*)(&oneConstant));
	//float test2 = *((float*)(&smallConstant));

	//float halfFloat = 0.5f;
	//float twoFloat = 2.0f;

	hkStringBuf sb; sb.printf( "input: 0x%x, 0x%x \n", *((int*)(&test1)), *((int*)(&test2)) );

	/*
	float intermediate = arg1 + arg2;
	intermediate -= arg3;
	*/

	float intermediate = 0.0f;
#ifndef _WIN64
	__asm
	{
		fld test2;
		//fmul halfFloat;
		fadd test1;
		fsub test1;
		//fmul twoFloat;
		fstp intermediate;
	}
#endif

	return intermediate;
}

double hkFpOpsTester::unregistered_cancelOpDenormDouble(double mulFactor)
{	
	double intermediate = 0.0;
#ifndef _WIN64
	long long arg1Int = 0x0000000000000000; 
	long long arg2Int = 0x0000000000000001; 

	double arg1 = *((double*)(&arg1Int));
	double arg2 = *((double*)(&arg2Int));

	//hkStringBuf sb; sb.printf( "input: 0x%x, 0x%x \n", *((int*)(&test1)), *((int*)(&test2)) );

	/*
	float intermediate = arg1 + arg2;
	intermediate -= arg3;
	*/
	
	__asm
	{
		fld arg2;
		fdiv mulFactor;
		fadd arg1;
		fsub arg1;
		fmul mulFactor;
		fstp intermediate;
	}
#endif


	return intermediate;
}

#ifdef ENABLE_SIMD_TEST
float hkFpOpsTester::unregistered_cancelOpSimd()
{
	int smallConstant = 0x33800000;
	float test1 = 1.0f;
	float test2 = *((float*)(&smallConstant));
	hkSimdReal test1Simd; test1Simd.setFromFloat(test1);
	hkSimdReal test2Simd; test2Simd.setFromFloat(test2);


	hkStringBuf sb; sb.printf( "input: 0x%x, 0x%x \n", *((int*)(&test1)), *((int*)(&test2)) );

	
	float intermediate;
	
	hkSimdReal intermediateSimd; intermediateSimd.setAdd(test1Simd, test2Simd);
	intermediateSimd = intermediateSimd - test1Simd;

	/*
	__asm
	{
		movss xmm0, test1;
		movss xmm1, test2;
		movss xmm3, xmm0;
		addss xmm3, xmm1;
		subss xmm3, xmm0
		movss intermediate, xmm3;
	}
	*/
	
	intermediate = (float)intermediateSimd.getReal();

	return intermediate;
}
#endif

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */

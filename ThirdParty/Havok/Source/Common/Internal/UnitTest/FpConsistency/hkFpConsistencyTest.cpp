/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Internal/hkInternal.h>

#include <Common/Base/UnitTest/hkUnitTest.h>


#if (defined( HK_PLATFORM_WIN32 ) || defined( HK_PLATFORM_X64 )) && !defined(HK_ARCH_ARM)

#include <Common/Internal/UnitTest/FpConsistency/hkFpOpsTester.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Container/String/hkStringBuf.h>

// This unit test performs series of fp and simd instructions on selected input variables, and keeps a crc of the outcomes which can be used to verify
// whether the operations perform equally on all (x86) platforms.
// The tested operations are, for fp: add, sub, mul, div, sqrt, sin, cos, tan, log and for simd: add, sub, mul, div, sqrt.
// Some tests try to find problems at specially constructed boundary cases (special inputs/outputs or rounding/register side-effects).
// Other tests simply probe certain domains by generating random inputs for that domain.
// - FP_TEST_SMALL enables the test suite to run in a limited amount of time as a unit test, but tests only a small number of random cases on top of the 
// complete set of specially constructed boundary cases. So it is not advised for serious testing.
// - FULL_ACC_RESULTS can be enabled in case the test is compiled with HK_ACC_FULL for every instruction in hkSseSimdReal.inl
// Note that it only changes the expected outcome of the full test, by itself it does *not* change the accuracy of the instructions, 
// you have to do that manually in the aforementioned file.

#define DISABLED_FOR_HAVOK
#define FP_TEST_SMALL
//#define FULL_ACC_RESULTS

#if defined HK_COMPILER_MSVC
//div instruction may be inaccurate on some pentiums,
//but that is the goal of the test
#pragma warning(disable: 4725)
#endif

typedef unsigned int uint32_t;

const uint32_t zeroInt = 0x00000000;
const uint32_t negZeroInt = 0x80000000;
const uint32_t infInt = 0x7f800000;
const uint32_t negInfInt = 0xff800000;
const uint32_t nanIntMax = 0x7fffffff;
const uint32_t nanIntMin = 0x7f800001;
const uint32_t negNanIntMin = 0xffffffff;
const uint32_t negNanIntMax = 0xff800001;
const uint32_t denormIntMin = 0x00000001;
const uint32_t denormIntMax = 0x007FFFFF;
const uint32_t negDenormIntMin = 0x807FFFFF;
const uint32_t negDenormIntMax = 0x80000001;

#ifdef FP_TEST_SMALL
const uint32_t NUM_OPWIDTH_TESTS = 10;
const uint32_t NUM_REGWIDTH_TESTS = 10;
const uint32_t NUM_SPECIAL_TESTS = 10;
const uint32_t NUM_SPECIAL_INFNAN_TESTS = 10;
const uint32_t NUM_ROUND_MODE_TESTS = 10;
const uint32_t NUM_ROUND_MODE_DENORM_TESTS = 10;
const uint32_t NUM_ADDITIONAL_ROUNDING_TESTS = 10;
const uint32_t NUM_SELECTIVE_DENORM_TESTS = 10;
const uint32_t NUM_SELECTIVE_INF_TESTS = 10;
const uint32_t NUM_DENORM_TESTS = 10;
const uint32_t NUM_DENORM_NORM_TESTS = 10;
const uint32_t NUM_RANDOM_TESTS = 10;
const uint32_t NUM_RANDOM_TRANSCENDENTAL_TESTS = 10;
#else
const uint32_t NUM_OPWIDTH_TESTS = 1000000;
const uint32_t NUM_REGWIDTH_TESTS = 1000000;
const uint32_t NUM_SPECIAL_TESTS = 5000000;
const uint32_t NUM_SPECIAL_INFNAN_TESTS = 5000000;
const uint32_t NUM_ROUND_MODE_TESTS = 100;
const uint32_t NUM_ROUND_MODE_DENORM_TESTS = 100;
const uint32_t NUM_ADDITIONAL_ROUNDING_TESTS = 1000000;
const uint32_t NUM_SELECTIVE_DENORM_TESTS = 1000000;
const uint32_t NUM_SELECTIVE_INF_TESTS = 1000000;
const uint32_t NUM_DENORM_TESTS = 1000000;
const uint32_t NUM_DENORM_NORM_TESTS = 1000000;
const uint32_t NUM_RANDOM_TESTS = 1000000000;
const uint32_t NUM_RANDOM_TRANSCENDENTAL_TESTS = 0xFFFFFFFF;
#endif

void printResult(float fpResult)
{
	hkStringBuf sb; sb.printf("Result: 0x%x, %f \n", *((uint32_t*)(&fpResult)), fpResult);
}

void printResult2(float fpResult, float fpResult2)
{
	hkStringBuf sb; 
	sb.printf("Result1: 0x%x, %.7f \n", *((uint32_t*)(&fpResult)), fpResult);
	sb.printf("Result2: 0x%x, %.7f \n", *((uint32_t*)(&fpResult2)), fpResult2);
}

float getRandRealFullRange(hkPseudoRandomGenerator randomGen)
{	
	const int v = randomGen.getRand32();
	return *((float*)(&v));
}

void showCpuInfo()
{
#ifndef DISABLED_FOR_HAVOK
	const char* szFeatures[] =
	{
		"x87 FPU On Chip",
		"Virtual-8086 Mode Enhancement",
		"Debugging Extensions",
		"Page Size Extensions",
		"Time Stamp Counter",
		"RDMSR and WRMSR Support",
		"Physical Address Extensions",
		"Machine Check Exception",
		"CMPXCHG8B Instruction",
		"APIC On Chip",
		"Unknown1",
		"SYSENTER and SYSEXIT",
		"Memory Type Range Registers",
		"PTE Global Bit",
		"Machine Check Architecture",
		"Conditional Move/Compare Instruction",
		"Page Attribute Table",
		"Page Size Extension",
		"Processor Serial Number",
		"CFLUSH Extension",
		"Unknown2",
		"Debug Store",
		"Thermal Monitor and Clock Ctrl",
		"MMX Technology",
		"FXSAVE/FXRSTOR",
		"SSE Extensions",
		"SSE2 Extensions",
		"Self Snoop",
		"Hyper-threading Technology",
		"Thermal Monitor",
		"Unknown4",
		"Pend. Brk. EN."
	};

	char CPUString[0x20];
	char CPUBrandString[0x40];
	int CPUInfo[4] = {-1};
	int nSteppingID = 0;
	int nModel = 0;
	int nFamily = 0;
	int nProcessorType = 0;
	int nExtendedmodel = 0;
	int nExtendedfamily = 0;
	int nBrandIndex = 0;
	int nCLFLUSHcachelinesize = 0;
	int nAPICPhysicalID = 0;
	int nFeatureInfo = 0;
	int nCacheLineSize = 0;
	int nL2Associativity = 0;
	int nCacheSizeK = 0;
	int nRet = 0;
	unsigned    nIds, nExIds, i;
	bool    bSSE3NewInstructions = false;
	bool    bMONITOR_MWAIT = false;
	bool    bCPLQualifiedDebugStore = false;
	bool    bThermalMonitor2 = false;


	// __cpuid with an InfoType argument of 0 returns the number of
	// valid Ids in CPUInfo[0] and the CPU identification string in
	// the other three array elements. The CPU identification string is
	// not in linear order. The code below arranges the information 
	// in a human readable form.
	__cpuid(CPUInfo, 0);
	nIds = CPUInfo[0];
	memset(CPUString, 0, sizeof(CPUString));
	*((int*)CPUString) = CPUInfo[1];
	*((int*)(CPUString+4)) = CPUInfo[3];
	*((int*)(CPUString+8)) = CPUInfo[2];

	// Get the information associated with each valid Id
	for (i=0; i<=nIds; ++i)
	{
		__cpuid(CPUInfo, i);
		//hkStringBuf sb; sb.printf_s("\nFor InfoType %d\n", i); 
		//hkStringBuf sb; sb.printf_s("CPUInfo[0] = 0x%x\n", CPUInfo[0]);
		//hkStringBuf sb; sb.printf_s("CPUInfo[1] = 0x%x\n", CPUInfo[1]);
		//hkStringBuf sb; sb.printf_s("CPUInfo[2] = 0x%x\n", CPUInfo[2]);
		//hkStringBuf sb; sb.printf_s("CPUInfo[3] = 0x%x\n", CPUInfo[3]);

		// Interpret CPU feature information.
		if  (i == 1)
		{
			nSteppingID = CPUInfo[0] & 0xf;
			nModel = (CPUInfo[0] >> 4) & 0xf;
			nFamily = (CPUInfo[0] >> 8) & 0xf;
			nProcessorType = (CPUInfo[0] >> 12) & 0x3;
			nExtendedmodel = (CPUInfo[0] >> 16) & 0xf;
			nExtendedfamily = (CPUInfo[0] >> 20) & 0xff;
			nBrandIndex = CPUInfo[1] & 0xff;
			nCLFLUSHcachelinesize = ((CPUInfo[1] >> 8) & 0xff) * 8;
			nAPICPhysicalID = (CPUInfo[1] >> 24) & 0xff;
			bSSE3NewInstructions = (CPUInfo[2] & 0x1) || false;
			bMONITOR_MWAIT = (CPUInfo[2] & 0x8) || false;
			bCPLQualifiedDebugStore = (CPUInfo[2] & 0x10) || false;
			bThermalMonitor2 = (CPUInfo[2] & 0x100) || false;
			nFeatureInfo = CPUInfo[3];
		}
	}

	
	// Calling __cpuid with 0x80000000 as the InfoType argument
	// gets the number of valid extended IDs.
	__cpuid(CPUInfo, 0x80000000);
	nExIds = CPUInfo[0];
	memset(CPUBrandString, 0, sizeof(CPUBrandString));

	
	// Get the information associated with each extended ID.
	for (i=0x80000000; i<=nExIds; ++i)
	{
		__cpuid(CPUInfo, i);
		//hkStringBuf sb; sb.printf_s("\nFor InfoType %x\n", i); 
		//hkStringBuf sb; sb.printf_s("CPUInfo[0] = 0x%x\n", CPUInfo[0]);
		//hkStringBuf sb; sb.printf_s("CPUInfo[1] = 0x%x\n", CPUInfo[1]);
		//hkStringBuf sb; sb.printf_s("CPUInfo[2] = 0x%x\n", CPUInfo[2]);
		//hkStringBuf sb; sb.printf_s("CPUInfo[3] = 0x%x\n", CPUInfo[3]);

		// Interpret CPU brand string and cache information.
		if  (i == 0x80000002)
			memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
		else if  (i == 0x80000003)
			memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
		else if  (i == 0x80000004)
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
		else if  (i == 0x80000006)
		{
			nCacheLineSize = CPUInfo[2] & 0xff;
			nL2Associativity = (CPUInfo[2] >> 12) & 0xf;
			nCacheSizeK = (CPUInfo[2] >> 16) & 0xffff;
		}
	}
	
	// Display all the information in user-friendly format.

	hkStringBuf sb; sb.printf_s("\n\nCPU String: %s\n", CPUString);

	if  (nIds >= 1)
	{
		if  (nSteppingID)
			hkStringBuf sb; sb.printf_s("Stepping ID = %d\n", nSteppingID);
		if  (nModel)
			hkStringBuf sb; sb.printf_s("Model = %d\n", nModel);
		if  (nFamily)
			hkStringBuf sb; sb.printf_s("Family = %d\n", nFamily);
		if  (nProcessorType)
			hkStringBuf sb; sb.printf_s("Processor Type = %d\n", nProcessorType);
		if  (nExtendedmodel)
			hkStringBuf sb; sb.printf_s("Extended model = %d\n", nExtendedmodel);
		if  (nExtendedfamily)
			hkStringBuf sb; sb.printf_s("Extended family = %d\n", nExtendedfamily);
		if  (nBrandIndex)
			hkStringBuf sb; sb.printf_s("Brand Index = %d\n", nBrandIndex);
		if  (nCLFLUSHcachelinesize)
			hkStringBuf sb; sb.printf_s("CLFLUSH cache line size = %d\n",
			nCLFLUSHcachelinesize);
		if  (nAPICPhysicalID)
			hkStringBuf sb; sb.printf_s("APIC Physical ID = %d\n", nAPICPhysicalID);

		if  (nFeatureInfo || bSSE3NewInstructions ||
			bMONITOR_MWAIT || bCPLQualifiedDebugStore ||
			bThermalMonitor2)
		{
			hkStringBuf sb; sb.printf_s("\nThe following features are supported:\n");

			if  (bSSE3NewInstructions)
				hkStringBuf sb; sb.printf_s("\tSSE3 New Instructions\n");
			if  (bMONITOR_MWAIT)
				hkStringBuf sb; sb.printf_s("\tMONITOR/MWAIT\n");
			if  (bCPLQualifiedDebugStore)
				hkStringBuf sb; sb.printf_s("\tCPL Qualified Debug Store\n");
			if  (bThermalMonitor2)
				hkStringBuf sb; sb.printf_s("\tThermal Monitor 2\n");

			
			i = 0;
			nIds = 1;
			while (i < (sizeof(szFeatures)/sizeof(const char*)))
			{
				if  (nFeatureInfo & nIds)
				{
					hkStringBuf sb; sb.printf_s("\t");
					hkStringBuf sb; sb.printf_s(szFeatures[i]);
					hkStringBuf sb; sb.printf_s("\n");
				}

				nIds <<= 1;
				++i;
			}
			
		}
	}

	if  (nExIds >= 0x80000004)
		hkStringBuf sb; sb.printf_s("\nCPU Brand String: %s\n", CPUBrandString);

	if  (nExIds >= 0x80000006)
	{
		hkStringBuf sb; sb.printf_s("Cache Line Size = %d\n", nCacheLineSize);
		hkStringBuf sb; sb.printf_s("L2 Associativity = %d\n", nL2Associativity);
		hkStringBuf sb; sb.printf_s("Cache Size = %dK\n", nCacheSizeK);
	}

	hkStringBuf sb; sb.printf_s("---------------------------------\n\n\n");
#endif
}

bool testSse2()
{
#ifndef DISABLED_FOR_HAVOK
#ifndef _WIN64
	__try 
	{
		__asm 
		{
			xorpd xmm0, xmm0        
		}
	}
	__except(EXCEPTION_EXECUTE_HANDLER) 
	{
		hkStringBuf sb; sb.printf("SSE2 Not supported.\n");
		return false;
	}
#endif
#endif
	return true;
}



float unregistered_cancelOpSimd()
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

void showRegWidth(hkFpOpsTester& opsTester)
{
	//int smallConstant = 0x26901D7D;

	//float arg1 = 1.0f;
	//float arg2 = *((float*)(&smallConstant));
	//float arg3 = 1.0f;

	float intermediate = opsTester.unregistered_cancelOp(/*arg1, arg2, arg2*/);
	float intermediateSimd = opsTester.unregistered_cancelOpSimd();

	printResult(intermediate);
	printResult(intermediateSimd);
}

#if 0
void showRegWidth()
{
	float fpResult;

	float arg1 = 1.0f;
	float arg2 = 1e-15f;
	float arg3 = 1e13f;

	float intermediate = cancelOp(arg1, arg2, arg3, arg3);

	fpResult = intermediate;
	printResult(fpResult);
}

void showRoundingMode()
{
	float fpResult;

	for(uint32_t i = 0; i < 4; ++i)
	{
		for(uint32_t j = 0; j < 0xF; ++j)
		{
			uint32_t arg1Int = 0x3f800000 + i;
			uint32_t arg2Int = 0x3e800000 + j;

			float arg1 = *( (float*)(&arg1Int) );
			float arg2 = *( (float*)(&arg2Int) );

			float intermediate = testAdd(arg1, arg2);

			fpResult = intermediate;
			printResult(fpResult);
		}
	}
}

void showSpecialValues()
{
	// These should not hold, but /fp:fast may force this behaviour anyway: x-x => 0, x*0 => 0, x-0 => x, x+0 => x, and 0-x => -x
	float arg[6] = { 10.0f, *((float*)(&infInt)), *((float*)(&nanIntMax)), *((float*)(&nanIntMin)), *((float*)(&negNanIntMin)), *((float*)(&negNanIntMax)) };
	float fZero = 0.0f;
	{
		hkStringBuf sb; sb.printf("Special test x-x => 0\n");
		for (int i = 0; i < 6; ++i)
		{
			float fpResult = testSub(arg[i], arg[i]);
			float fpResult2 = arg[i] - arg[i];
			printResult2(fpResult,fpResult2);
		}	

		hkStringBuf sb; sb.printf("Special test x*0 => 0\n");
		for (int i = 0; i < 6; ++i)
		{
			float fpResult = testMul(arg[i], fZero);
			float fpResult2 = arg[i] * fZero;
			printResult2(fpResult,fpResult2);
		}	

		hkStringBuf sb; sb.printf("Special test x-0 => x\n");
		for (int i = 0; i < 6; ++i)
		{
			float fpResult = testSub(arg[i], fZero);
			float fpResult2 = arg[i] - fZero;
			printResult2(fpResult,fpResult2);
		}	

		hkStringBuf sb; sb.printf("Special test x+0 => x\n");
		for (int i = 0; i < 6; ++i)
		{
			float fpResult = testAdd(arg[i], fZero);
			float fpResult2 = arg[i] + fZero;
			printResult2(fpResult,fpResult2);
		}	

		hkStringBuf sb; sb.printf("Special test 0-x => -x\n");
		for (int i = 0; i < 6; ++i)
		{
			float fpResult = testSub(fZero, arg[i]);
			float fpResult2 = fZero - arg[i];
			printResult2(fpResult,fpResult2);
		}	
	}

	// These should hold: oo + oo == oo, -oo - oo == -oo, oo - oo == NaN, oo * oo == oo, 0 * oo == NaN, oo / oo == NaN, 0 / oo == 0, oo / 0 == oo, 0 / 0 == NaN,
	{
		hkStringBuf sb; sb.printf("Special test oo + oo == oo\n");
		{
			float fpResult = testAdd(arg[1], arg[1]);
			float fpResult2 = arg[1] + arg[1];
			printResult2(fpResult,fpResult2);
		}	
		hkStringBuf sb; sb.printf("Special test -oo - oo == -oo\n");
		{
			float fpResult = testSub(-arg[1], arg[1]);
			float fpResult2 = -arg[1] - arg[1];
			printResult2(fpResult,fpResult2);
		}	
		hkStringBuf sb; sb.printf("Special test oo - oo == NaN\n");
		{
			float fpResult = testSub(arg[1], arg[1]);
			float fpResult2 = arg[1] - arg[1];
			printResult2(fpResult,fpResult2);
		}	
		hkStringBuf sb; sb.printf("Special test oo * oo == oo\n");
		{
			float fpResult = testMul(arg[1], arg[1]);
			float fpResult2 = arg[1] * arg[1];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test 0 * oo == NaN\n");
		{
			float fpResult = testMul(fZero, arg[1]);
			float fpResult2 = fZero * arg[1];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test oo / oo == NaN\n");
		{
			float fpResult = testDiv(arg[1], arg[1]);
			float fpResult2 = arg[1] / arg[1];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test 0 / oo == 0\n");
		{
			float fpResult = testDiv(fZero, arg[1]);
			float fpResult2 = fZero / arg[1];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test oo / 0 == oo\n");
		{
			float fpResult = testDiv(arg[1], fZero);
			float fpResult2 = arg[1] / fZero;
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test 0 / 0 == NaN\n");
		{
			float fpResult = testDiv(fZero, fZero);
			float fpResult2 = fZero / fZero;
			printResult2(fpResult,fpResult2);
		}	
	}

	// And these as well: x + NaN == NaN, x - NaN == NaN, x * NaN == NaN, x / NaN == NaN, x + oo == oo, x - oo == -oo, x * oo == oo, x / oo == 0 where x != NaN and x is not such that the rule appears in the previous comment
	{
		hkStringBuf sb; sb.printf("Special test x + NaN == NaN \n");	
		{
			float fpResult = testAdd(arg[0], arg[2]);
			float fpResult2 = arg[0] + arg[2];
			printResult2(fpResult,fpResult2);
		}
		{
			float fpResult = testAdd(arg[0], arg[4]);
			float fpResult2 = arg[0] + arg[4];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test x - NaN == NaN \n");	
		{
			float fpResult = testSub(arg[0], arg[2]);
			float fpResult2 = arg[0] - arg[2];
			printResult2(fpResult,fpResult2);
		}
		{
			float fpResult = testSub(arg[0], arg[4]);
			float fpResult2 = arg[0] - arg[4];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test x * NaN == NaN \n");	
		{
			float fpResult = testMul(arg[0], arg[2]);
			float fpResult2 = arg[0] * arg[2];
			printResult2(fpResult,fpResult2);
		}
		{
			float fpResult = testMul(arg[0], arg[4]);
			float fpResult2 = arg[0] * arg[4];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test x / NaN == NaN \n");	
		{
			float fpResult = testDiv(arg[0], arg[2]);
			float fpResult2 = arg[0] / arg[2];
			printResult2(fpResult,fpResult2);
		}
		{
			float fpResult = testDiv(arg[0], arg[4]);
			float fpResult2 = arg[0] / arg[4];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test x + oo == oo \n");	
		{
			float fpResult = testAdd(arg[0], arg[1]);
			float fpResult2 = arg[0] + arg[2];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test x - oo == -oo \n");	
		{
			float fpResult = testSub(arg[0], arg[1]);
			float fpResult2 = arg[0] - arg[2];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test x * oo == oo \n");	
		{
			float fpResult = testMul(arg[0], arg[1]);
			float fpResult2 = arg[0] * arg[2];
			printResult2(fpResult,fpResult2);
		}
		hkStringBuf sb; sb.printf("Special test x / oo == 0 \n");	
		{
			float fpResult = testDiv(arg[0], arg[1]);
			float fpResult2 = arg[0] / arg[2];
			printResult2(fpResult,fpResult2);
		}

	}

	// Some extra checks for NaN
	{
		hkStringBuf sb; sb.printf("Extra NaN tests (probably superfluous)\n");	
		{
			float fpResult = testAdd(arg[2], arg[2]);
			float fpResult2 = arg[2] + arg[2];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testAdd(arg[2], arg[4]);
			float fpResult2 = arg[2] + arg[4];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testAdd(arg[4], arg[4]);
			float fpResult2 = arg[4] + arg[4];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testSub(arg[2], arg[2]);
			float fpResult2 = arg[2] - arg[2];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testSub(arg[2], arg[4]);
			float fpResult2 = arg[2] - arg[4];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testSub(arg[4], arg[4]);
			float fpResult2 = arg[4] - arg[4];
			printResult2(fpResult,fpResult2);
		}
		{
			float fpResult = testMul(arg[2], arg[2]);
			float fpResult2 = arg[2] * arg[2];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testMul(arg[2], arg[4]);
			float fpResult2 = arg[2] * arg[4];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testMul(arg[4], arg[4]);
			float fpResult2 = arg[4] * arg[4];
			printResult2(fpResult,fpResult2);
		}
		{
			float fpResult = testDiv(arg[2], arg[2]);
			float fpResult2 = arg[2] / arg[2];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testDiv(arg[2], arg[4]);
			float fpResult2 = arg[2] / arg[4];
			printResult2(fpResult,fpResult2);
		}	
		{
			float fpResult = testDiv(arg[4], arg[4]);
			float fpResult2 = arg[4] / arg[4];
			printResult2(fpResult,fpResult2);
		}
	
	}
}
#endif


void testOpWidth(hkFpOpsTester& opsTester, hkPseudoRandomGenerator& randomGen)
{
	//test how far away a non-representable mantissa bit can be of influence on the end result for double computations,
	//using round-to-nearest-even to distinguish two different rounding cases.
	{
		long long arg1Int = 0x3ff0000000000000;
		long long arg2Int = 0x3ca0000000000000;

		double arg1 = *((double*)(&arg1Int));
		double arg2 = *((double*)(&arg2Int));

		opsTester.testAddDouble(arg1, arg2); //the result is exactly halfway (only the first non representable bit is set), so rounds down to 0
		opsTester.testAddDoubleSimd(arg1, arg2);
	}

	{
		long long arg1Int = 0x3ff0000000000000;
		long long arg2Int = 0x3ca0000000000001;

		double arg1 = *((double*)(&arg1Int));
		double arg2 = *((double*)(&arg2Int));

		opsTester.testAddDouble(arg1, arg2); //now the non-representable part of the result has the first and the last bit set; in this case it has to round up to 1
		opsTester.testAddDoubleSimd(arg1, arg2);
	}

	//we generate four similar cases, now for testing multiplication
	{
		long long arg1Int = 0x3ff0000000000001; //lsb of mantissa set 
		long long arg2Int = 0x3ff8000000000000; //msb of mantissa set (not counting implicit bit)

		double arg1 = *((double*)(&arg1Int));
		double arg2 = *((double*)(&arg2Int));

		opsTester.testMulDouble(arg1, arg2); //exact mantissa result should be (msb)1000...0001(lsb) followed by 1 (not representable). Should be rounded to (msb)1000...0010(lsb).
		opsTester.testMulDoubleSimd(arg1, arg2);
	}

	{
		long long arg1Int = 0x3ff0000000000003; //two lsbs of mantissa set 
		long long arg2Int = 0x3ff8000000000001; //msb and lsb of mantissa set (not counting implicit bit)

		double arg1 = *((double*)(&arg1Int));
		double arg2 = *((double*)(&arg2Int));

		opsTester.testMulDouble(arg1, arg2); //exact mantissa result should be (msb)1000...0101(lsb) followed by 1000...0011 (not representable). Should be rounded to (msb)1000...0110(lsb).
		opsTester.testMulDoubleSimd(arg1, arg2);
	}

	{
		long long arg1Int = 0x3ff0000000000003; //two lsbs of mantissa set 
		long long arg2Int = 0x3ff8000000000000; //msb of mantissa set (not counting implicit bit)

		double arg1 = *((double*)(&arg1Int));
		double arg2 = *((double*)(&arg2Int));

		opsTester.testMulDouble(arg1, arg2); //exact mantissa result should be (msb)1000...0100(lsb) followed by 1 (not representable). Should be rounded to (msb)1000...0100(lsb).
		opsTester.testMulDoubleSimd(arg1, arg2);
	}

	{
		long long arg1Int = 0x3ff0000000000001; //lsb of mantissa set
		long long arg2Int = 0x3ff8000000000001; //msb and lsb of mantissa set (not counting implicit bit)

		double arg1 = *((double*)(&arg1Int));
		double arg2 = *((double*)(&arg2Int));

		opsTester.testMulDouble(arg1, arg2); //exact mantissa result should be (msb)1000...0010(lsb) followed by 1000...0011 (not representable). Should be rounded to (msb)1000...0011(lsb).
		opsTester.testMulDoubleSimd(arg1, arg2);
	}
	
	{
		hkStringBuf sb; sb.printf("Op width test part one - Fp: 0x%x, FpDiv: 0x%x, Simd: 0x%x, SimdDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd(), opsTester.getCrcSimdDiv());
	}

	//test the width of operation accuracy
	for(uint32_t i = 0; i < NUM_OPWIDTH_TESTS; ++i)
	{
		hkFpOpsTester::DoubleUnion arg1, arg2;
		
		arg1.m_int32[0] = randomGen.getRand32();
		arg1.m_int32[1] = randomGen.getRand32();
		arg2.m_int32[0] = randomGen.getRand32();
		arg2.m_int32[1] = randomGen.getRand32();

		opsTester.testAddDouble(arg1.m_float64, arg2.m_float64);
		opsTester.testAddDoubleSimd(arg1.m_float64, arg2.m_float64);
		opsTester.testMulDouble(arg1.m_float64, arg2.m_float64);
		opsTester.testMulDoubleSimd(arg1.m_float64, arg2.m_float64);
		opsTester.testDivDouble(arg1.m_float64, arg2.m_float64);
		opsTester.testDivDoubleSimd(arg1.m_float64, arg2.m_float64);
	}

	{
		hkStringBuf sb; sb.printf("Op width test finished with crcs - Fp: 0x%x, FpDiv: 0x%x, Simd: 0x%x, SimdDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd(), opsTester.getCrcSimdDiv());
	}
}

void testRegWidth(hkFpOpsTester& opsTester, hkPseudoRandomGenerator& randomGen)
{
	float arg1 = 1.0f;
	float arg2 = 1e-15f;
	float arg3 = 1.0f;//1e13f;

	{
		opsTester.cancelOpFp(arg1, arg2, arg3, arg3);
		opsTester.cancelOpSimd(arg1, arg2, arg3, arg3);
	}
	
	for(uint32_t i = 0; i < NUM_REGWIDTH_TESTS; ++i)
	{
		float x = getRandRealFullRange(randomGen);

		opsTester.cancelOpFp(x, arg2*x, arg3, x*arg3);
		opsTester.cancelOpSimd(x, arg2*x, arg3, x*arg3);
	}

#ifndef _WIN64
	//multiply the smallest possible value possibly held by a register by the widest mantissa provided by the enabled precision (24,53 or 64 bit)
	{
		//uint32_t smallestDenormInt = 0x1;
		uint32_t ultimateSmallestNumberInt = 0x3;
		uint32_t oneHalfInt = 0x3fc00000;
		uint32_t oneHalfPlusEpsilonInt = 0x3fc00001;

		//float smallestDenorm = *((float*)(&smallestDenormInt));
		float ultimateSmallestNumber = *((float*)(&ultimateSmallestNumberInt));
		float oneHalf = *((float*)(&oneHalfInt));
		float oneHalfPlusEpsilon = *((float*)(&oneHalfPlusEpsilonInt));

		float halfFloat = 0.5f;

		//minimum exponents for various precisions
		//32: -127-22=-149
		//64: 
		//79: -16383-22=-16405
		//80: -32767-22=-32789
		for( uint32_t exponentTest = 0; exponentTest < 4; ++exponentTest)
		{
			float resultFp, resultOneHalfPlusE, resultOneHalf, resultSimd, resultSimdOneHalf, resultSimdOneHalfPlusE;

			__asm
			{
				fld ultimateSmallestNumber;	
			}

			//16405-149
			for(uint32_t i = 0; i < 16256+exponentTest; ++i)
			{
				__asm
				{
					fmul halfFloat; 	
				}	
			}

			__asm
			{
				fld st;
				fld st; //
				fld oneHalf;
				fmul st(2), st(0);
				fstp st;
				fld oneHalfPlusEpsilon; //
				fmul st(1), st(0); //
				fstp st; //
			}
			

			for(uint32_t i = 0; i < 16256+exponentTest; ++i)
			{
				__asm
				{
					fld halfFloat;
					fdiv st(1), st(0); 
					fdiv st(2), st(0); 	
					fdiv st(3), st(0); 	
					fstp st;
				}	
			}

			__asm
			{
				fstp resultOneHalfPlusE;
				fstp resultOneHalf;
				fstp resultFp;
			}

			//for simd, one is enough to kill it
			__asm
			{
				movss xmm0, ultimateSmallestNumber;	
			}

			for(uint32_t i = 0; i < exponentTest; ++i)
			{
				__asm
				{
					mulss xmm0, halfFloat; 	
				}	
			}

			__asm
			{
				movss xmm1, xmm0;
				movss xmm2, xmm0;
				mulss xmm1, oneHalf;
				mulss xmm2, oneHalfPlusEpsilon;
			}

			for(uint32_t i = 0; i < exponentTest; ++i)
			{
				__asm
				{
					divss xmm0, halfFloat; 
					divss xmm1, halfFloat;
					divss xmm2, halfFloat;
				}	
			}

			__asm
			{
				movss resultSimd, xmm0;
				movss resultSimdOneHalf, xmm1;
				movss resultSimdOneHalfPlusE, xmm2;
			}

			opsTester.testAddFp(resultFp, 0.0f);
			opsTester.testAddSimd(resultSimd, 0.0f);
			//hkStringBuf sb; sb.printf("Ultimate regwidth test result fp: 0x%x, simd: 0x%x \n", *((int*)(&resultFp)), *((int*)(&resultSimd)));
			//hkStringBuf sb; sb.printf("Ultimate regwidth test result OneHalf fp: 0x%x, simd: 0x%x \n", *((int*)(&resultOneHalf)), *((int*)(&resultSimdOneHalf)));
			//hkStringBuf sb; sb.printf("Ultimate regwidth test result OneHalfPlusE fp: 0x%x, simd: 0x%x \n", *((int*)(&resultOneHalfPlusE)), *((int*)(&resultSimdOneHalfPlusE)));
		}
	}
#endif

	hkStringBuf sb; sb.printf("Register width test finished with crcs - Fp: 0x%x, FpDiv: 0x%x, Simd: 0x%x, SimdDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd(), opsTester.getCrcSimdDiv());
}

void testRegWidthDouble(hkFpOpsTester& opsTester, hkPseudoRandomGenerator& randomGen)
{
	{
		double mulFactor = 1.0;
		double zero = 0.0;
		for(uint32_t i = 0; i < 0x03FF; ++i, mulFactor *= 2.0f)
		{
			hkFpOpsTester::DoubleUnion denormResult;
			denormResult.m_float64 = opsTester.unregistered_cancelOpDenormDouble(mulFactor);
			opsTester.testAddDouble(denormResult.m_float64, zero); //save the result
		}
	}

	hkStringBuf sb; sb.printf("Register double width test finished with crcs - Fp: 0x%x, FpDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv());
}

void testRegWidthDoubleExtended(hkFpOpsTester& opsTester)
{
#ifndef _WIN64
	{
		//the goal of the mess below is to multiply 1.100..001b^minexp with 1.00..001 in the range of a 79-bit float register,
		//and see whether it rounds correctly by taking into account the very last bit of the result (which is double the width of the input values.
		//The exact result should be 1.100..0010|100..001b^minexp where the symbol | signifies the limit of register precision, and should therefore round up instead of down
		//We check only the last bits of the result before the register limit, so the result should be 3)

		uint32_t ultimateSmallestNumberInt = 0x1;
		float ultimateSmallestNumber = *((float*)(&ultimateSmallestNumberInt));
		//float threeFloat = 3.0f;
		float halfFloat = 0.5f;
		float oneHalfFloat = 1.5f;

		float resultFp;

		__asm
		{
			fld ultimateSmallestNumber;	
		}
		//-16405+149-40 (+exp smallest single denorm - mantissa to extended)
		for(uint32_t i = 0; i < 16296; ++i)
		{
			__asm
			{
				fmul halfFloat; 	
			}	
		}

		//prepare two operands, 
		__asm
		{
			fld st;
			fld1;
		}

		for(uint32_t i = 0; i < 63; ++i)
		{
			__asm
			{
				fld halfFloat;
				fdiv st(2), st; //first operand
				fmul st(1), st; //second operand	
				fstp st;
			}	
		}

		//first operand st(1) is now the highest bit of the smallest value representable
		//second operand st(0) is the lsb of normalized value 1. Make it 11....1 by adding 1.5
		__asm
		{
			fld oneHalfFloat;
			fadd st(1), st(0);
		}

		__asm
		{
			fincstp;
				fincstp;
					fadd st(1), st(0); //combine lowest bit with highest bit -> first operand
				fdecstp;
				//(disregarding the remaining fincstp) st(0): 3, st(1): 11.....1, st(2): highest bit of smallest number, st(3): 1.....1		
				fmul st(2), st(0); //multiply first operand with the second operand
			fdecstp;

			fmul st(0), st(2); //calculate the highest bits of the previous result
			fsub st(3), st(0); //remove the highest bits from the result
			fstp st; //remove the top two 
			fstp st; 
			fstp st; 
		}

		for(uint32_t i = 0; i < 16296; ++i)
		{
			__asm
			{
				fdiv halfFloat; 	
			}	
		}
		__asm
		{
			fstp resultFp; //store result of mul
		}

		//hkStringBuf sb; sb.printf("Double Extended regwidth test special case result fp: 0x%x \n", *((int*)(&resultFp)));
		opsTester.testAddFp(resultFp,0.0f); //just add the check to both fp and simd
		opsTester.testAddSimd(resultFp,0.0f);
	}

	//multiply the smallest possible value possibly held by a register by the widest mantissa provided by the enabled precision (24,53 or 64 bit)
	{
		//uint32_t ultimateSmallestNumberInt = 0x1;
		//uint32_t halfPlusEpsilonInt = 0x3f000001;

		//float halfPlusEpsilon = *((float*)(&halfPlusEpsilonInt));

		float halfFloat = 0.5f;

		//minimum exponents for various precisions
		//32: -127-22=-149
		//64: 
		//79: -16383-22=-16405
		//80: -32767-22=-32789
		for( uint32_t exponentTest = 0; exponentTest < 6; ++exponentTest)
		{
			uint32_t ultimateSmallestNumberInt = exponentTest;
			float ultimateSmallestNumber = *((float*)(&ultimateSmallestNumberInt));

			float resultFp, resultOneHalfPlusE, resultOneHalf;

			__asm
			{
				fld ultimateSmallestNumber;	
			}

			//-16405+149-40 (+exp smallest single denorm - mantissa to extended)
			for(uint32_t i = 0; i < 16296; ++i)
			{
				__asm
				{
					fmul halfFloat; 	
				}	
			}

			//multiply with half
			__asm
			{
				fld st;
				fld halfFloat;
				fmul st(1), st(0);
				fstp st;
			}

			//multiply with half + 79-bit epsilon
			__asm
			{
				fld st(1);
				fld1;
			}

			//create the epsilon: mantissa of 63 bits, but we're adding to a half later
			for(uint32_t i = 0; i < 64; ++i)
			{
				__asm
				{
					fmul halfFloat; 	
				}	
			}

			//add to half and multiply with the double extended smallest number
			__asm
			{
				fadd halfFloat;
				fmul st(1), st(0);
				fstp st;
			}

			//get the results back to a normal range
			for(uint32_t i = 0; i < 16296; ++i)
			{
				__asm
				{
					fld halfFloat;
					fdiv st(1), st(0); 
					fdiv st(2), st(0); 	
					fdiv st(3), st(0); 	
					fstp st;
				}	
			}

			__asm
			{
				fstp resultOneHalfPlusE;
				fstp resultOneHalf;
				fstp resultFp;
			}

			//hkStringBuf sb; sb.printf("Double Extended regwidth test result fp: 0x%x \n", *((int*)(&resultFp)));
			//hkStringBuf sb; sb.printf("Double Extended regwidth test result OneHalf fp: 0x%x \n", *((int*)(&resultOneHalf)));
			//hkStringBuf sb; sb.printf("Double Extended regwidth test result OneHalfPlusE fp: 0x%x \n", *((int*)(&resultOneHalfPlusE)));
			opsTester.testAddFp(resultFp,0.0f); //just add the check to both fp and simd
			opsTester.testAddSimd(resultFp,0.0f);
			opsTester.testAddFp(resultOneHalf,0.0f); //just add the check to both fp and simd
			opsTester.testAddSimd(resultOneHalf,0.0f);
			opsTester.testAddFp(resultOneHalfPlusE,0.0f); //just add the check to both fp and simd
			opsTester.testAddSimd(resultOneHalfPlusE,0.0f);
		}
	}
#endif
	hkStringBuf sb; sb.printf("Double extended regwidth test finished with crc: 0x%x \n\n", opsTester.getCrcFp());

}

void testRoundingModeAndBoundaries(hkFpOpsTester& opsTester, hkPseudoRandomGenerator& randomGen)
{
	uint32_t roundingModeArg1[] = { 0x3f800000, 0xbf800000, zeroInt,		negZeroInt, zeroInt,	negZeroInt, denormIntMax,	negDenormIntMin,	denormIntMax,	negDenormIntMin,	denormIntMin,	negDenormIntMax, infInt,		negInfInt,	infInt,		negInfInt,	nanIntMin,	negNanIntMax,	nanIntMax,	negNanIntMin, 0x01000000, 0x81000000, 0x7f000000, 0xff000000 };
	uint32_t roundingModeArg2[] = { 0x3e800000, 0xbe800000, 0x00000001,		0x80000001, 0x80000001, 0x00000001, 0x00000001,		0x80000001,			0x80000001,		0x00000001,			0x80000001,		0x00000001,		 0xfe800001,	0x7e800001,	0x7e800001, 0xfe800001,	0xfe800001, 0x7e800001,		0x7e800001, 0xfe800001,   0x80000001, 0x00000001, 0x7e000001, 0xfe000001 };

	//rounding mode chosen cases
	for(uint32_t roundingCase = 0; roundingCase < sizeof(roundingModeArg1)/sizeof(uint32_t); ++roundingCase )
	{
		for(uint32_t i = 0; i < NUM_ROUND_MODE_TESTS; ++i)
		{
			for(uint32_t j = 0; j < 0xF; ++j)
			{
				uint32_t arg1Int = roundingModeArg1[roundingCase] + i;
				uint32_t arg2Int = roundingModeArg2[roundingCase] + j;

				float arg1 = *( (float*)(&arg1Int) );
				float arg2 = *( (float*)(&arg2Int) );

				opsTester.testAddFp(arg1, arg2);
				opsTester.testAddSimd(arg1, arg2);
			}
		}
	}

	//rounding mode with random numbers
	{
		uint32_t inRegularRangeCount = 0;
	
		for(uint32_t roundingCase = 0; roundingCase < NUM_ADDITIONAL_ROUNDING_TESTS; ++roundingCase )
		{
			float x = getRandRealFullRange(randomGen);
			inRegularRangeCount += (x > *((float*)(&denormIntMax))) && (x < *((float*)(&infInt))) || (x < *((float*)(&negDenormIntMin))) && (x > *((float*)(&negInfInt)));

			uint32_t baseInt = *((uint32_t*)(&x));
			//clear the mantissa, set its last bit, and divide by 4 to get the step
			uint32_t stepInt = (baseInt & 0xff800000) | 0x00000001; 
			float y = *((float*)(&stepInt)) * 0.25f;
			stepInt = *((uint32_t*)(&y));

			for(uint32_t i = 0; i < NUM_ROUND_MODE_TESTS; ++i)
			{
				for(uint32_t j = 0; j < 0xF; ++j)
				{
					uint32_t arg1Int = baseInt + i;
					uint32_t arg2Int = stepInt + j;

					float arg1 = *( (float*)(&arg1Int) );
					float arg2 = *( (float*)(&arg2Int) );

					opsTester.testAddFp(arg1, arg2);
					opsTester.testAddSimd(arg1, arg2);
				}
			}
		}

		hkStringBuf sb; sb.printf( "Rounding modes and boundaries - random values in normal range: %i \n", inRegularRangeCount );
	}

#ifndef _WIN64
	//rounding mode for denormalized numbers
	for(uint32_t i = 0; i < NUM_ROUND_MODE_DENORM_TESTS; ++i)
	{
		uint32_t denormInt = randomGen.getRand32();
		denormInt = denormInt & 0x807fffff;
		uint32_t smallestDenormInt = 0x6;
		float denorm = *((float*)(&denormInt));
		float smallestDenorm = *((float*)(&smallestDenormInt));
		float quarter = 0.25f;

		//hkStringBuf sb; sb.printf("Denorm 0x%x \n", denormInt);

		for(float i = 0.0f; i < 16.0f; i+=1.0f)
		{
			float resultFp;
			float resultSimd;
			__asm
			{
				fld smallestDenorm;
				fmul quarter;
				fmul i;
				fadd denorm;
				fstp resultFp;
			}
			__asm
			{
				movss xmm0, smallestDenorm;
				mulss xmm0, quarter;
				mulss xmm0, i;
				addss xmm0, denorm;
				movss resultSimd, xmm0;
			}
			//hkStringBuf sb; sb.printf("Result fp: 0x%x \n", *((uint32_t*)(&resultFp)));
			//hkStringBuf sb; sb.printf("Result simd: 0x%x \n", *((uint32_t*)(&resultSimd)));
			opsTester.testAddFp(resultFp, 0.0f);
			opsTester.testAddSimd(resultSimd, 0.0f);
		}
	}
#endif

	hkStringBuf sb; sb.printf("Rounding modes and boundaries test finished with crcs - Fp: 0x%x, FpDiv: 0x%x, Simd: 0x%x, SimdDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd(), opsTester.getCrcSimdDiv());
}

void testSpecialValues(hkFpOpsTester& opsTester, hkPseudoRandomGenerator& randomGen)
{
	uint32_t halfNaN = nanIntMin/2+nanIntMax/2;
	uint32_t halfNegNaN = negNanIntMin/2+negNanIntMax/2;
	float specials[] = { 1.0f, -1.0f, 10.0f, -10.0f, 
		*((float*)(&zeroInt)), *((float*)(&negZeroInt)),
		*((float*)(&infInt)), *((float*)(&negInfInt)), 
		*((float*)(&nanIntMin)), *((float*)(&nanIntMax)), *((float*)(&negNanIntMin)), *((float*)(&negNanIntMax)), 
		*((float*)(&halfNaN)), *((float*)(&halfNegNaN)),
		*((float*)(&denormIntMin)), *((float*)(&denormIntMax)), 
		*((float*)(&negDenormIntMin)), *((float*)(&negDenormIntMax)) };

	//test all special values against eachother
	for( int i = 0; i < sizeof(specials)/sizeof(float); ++i )
	{
		for( int j = 0; j < sizeof(specials)/sizeof(float); ++j )
		{
			opsTester.testAddFp(specials[i], specials[j]);
			opsTester.testSubFp(specials[i], specials[j]);
			opsTester.testDivFp(specials[i], specials[j]);
			opsTester.testMulFp(specials[i], specials[j]);
			opsTester.testAddSimd(specials[i], specials[j]);
			opsTester.testSubSimd(specials[i], specials[j]);
			opsTester.testDivSimd(specials[i], specials[j]);
			opsTester.testMulSimd(specials[i], specials[j]);
		}
	}

	//test special/normal value interaction for problematic cases
	{
		const int nanBaseIndex = 8;
		const int infBaseIndex = 6;	
		uint32_t inRegularRangeCount = 0;
	
		for(uint32_t i = 0; i < NUM_SPECIAL_TESTS; ++i)
		{
			float x = getRandRealFullRange(randomGen);
			inRegularRangeCount += (x > *((float*)(&denormIntMax))) && (x < *((float*)(&infInt))) || (x < *((float*)(&negDenormIntMin))) && (x > *((float*)(&negInfInt)));

			//hkStringBuf sb; sb.printf("Special test x + NaN == NaN \n");	
			for(int j = 0; j < 4; ++j)
			{
				opsTester.testAddFp(x, specials[nanBaseIndex+j]);
				opsTester.testAddFp(specials[nanBaseIndex+j],x);
				opsTester.testAddSimd(x, specials[nanBaseIndex+j]);
				opsTester.testAddSimd(specials[nanBaseIndex+j],x);
			}
			//hkStringBuf sb; sb.printf("Special test x - NaN == NaN \n");	
			for(int j = 0; j < 4; ++j)
			{
				opsTester.testSubFp(x, specials[nanBaseIndex+j]);
				opsTester.testSubFp(specials[nanBaseIndex+j], x);
				opsTester.testSubSimd(x, specials[nanBaseIndex+j]);
				opsTester.testSubSimd(specials[nanBaseIndex+j], x);
			}
			//hkStringBuf sb; sb.printf("Special test x * NaN == NaN \n");	
			for(int j = 0; j < 4; ++j)
			{
				opsTester.testMulFp(x, specials[nanBaseIndex+j]);
				opsTester.testMulFp(specials[nanBaseIndex+j], x);
				opsTester.testMulSimd(x, specials[nanBaseIndex+j]);
				opsTester.testMulSimd(specials[nanBaseIndex+j], x);
			}
			//hkStringBuf sb; sb.printf("Special test x / NaN == NaN \n");	
			for(int j = 0; j < 4; ++j)
			{
				opsTester.testDivFp(x, specials[nanBaseIndex+j]);
				opsTester.testDivFp(specials[nanBaseIndex+j], x);
				opsTester.testDivSimd(x, specials[nanBaseIndex+j]);
				opsTester.testDivSimd(specials[nanBaseIndex+j], x);
			}
			//hkStringBuf sb; sb.printf("Special test x + oo == oo \n");
			for(int j = 0; j < 2; ++j)
			{
				opsTester.testAddFp(x, specials[infBaseIndex+j]);
				opsTester.testAddFp(specials[infBaseIndex+j], x);
				opsTester.testAddSimd(x, specials[infBaseIndex+j]);
				opsTester.testAddSimd(specials[infBaseIndex+j], x);
			}
			//hkStringBuf sb; sb.printf("Special test x - oo == -oo \n");	
			for(int j = 0; j < 2; ++j)
			{
				opsTester.testSubFp(x, specials[infBaseIndex+j]);
				opsTester.testSubFp(specials[infBaseIndex+j], x);
				opsTester.testSubSimd(x, specials[infBaseIndex+j]);
				opsTester.testSubSimd(specials[infBaseIndex+j], x);
			}
			//hkStringBuf sb; sb.printf("Special test x * oo == oo \n");
			for(int j = 0; j < 2; ++j)
			{
				opsTester.testMulFp(x, specials[infBaseIndex+j]);
				opsTester.testMulFp(specials[infBaseIndex+j], x);
				opsTester.testMulSimd(x, specials[infBaseIndex+j]);
				opsTester.testMulSimd(specials[infBaseIndex+j], x);
			}
			//hkStringBuf sb; sb.printf("Special test x / oo == 0 \n");	
			for(int j = 0; j < 2; ++j)
			{
				opsTester.testDivFp(x, specials[infBaseIndex+j]);
				opsTester.testDivFp(specials[infBaseIndex+j], x);
				opsTester.testDivSimd(x, specials[infBaseIndex+j]);
				opsTester.testDivSimd(specials[infBaseIndex+j], x);
			}
		}


		hkStringBuf sb; sb.printf( "Special value test - random values in normal range: %i \n", inRegularRangeCount );
	}

	//test pairs of (x op y) with one operand in the inf..nan range and the other outside of that range
	{
		for(int i = 0; i < NUM_SPECIAL_INFNAN_TESTS; ++i)
		{
			uint32_t xInt = randomGen.getRand32();
			uint32_t yInt[2];
			yInt[0] = yInt[1] = randomGen.getRand32();

			yInt[0] |= 0x7f800000; //generate a NaN (pos or neg)
			yInt[1] = (yInt[1] | 0x7f800000) & 0xff800000; //generate an Inf (pos or neg)

			float x = *((float*)(&xInt));
			float y[2] = {*((float*)(&yInt[0])), *((float*)(&yInt[1]))};

			for(int j = 0; j < 2; ++j)
			{
				{
					opsTester.testAddFp(x,y[j]);
					opsTester.testAddFp(y[j],x);
					opsTester.testSubFp(x,y[j]);
					opsTester.testSubFp(y[j],x);
					opsTester.testMulFp(x,y[j]);
					opsTester.testMulFp(y[j],x);
					opsTester.testDivFp(x,y[j]);
					opsTester.testDivFp(y[j],x);
				}
				{
					opsTester.testAddSimd(x,y[j]);
					opsTester.testAddSimd(y[j],x);
					opsTester.testSubSimd(x,y[j]);
					opsTester.testSubSimd(y[j],x);
					opsTester.testMulSimd(x,y[j]);
					opsTester.testMulSimd(y[j],x);
					opsTester.testDivSimd(x,y[j]);
					opsTester.testDivSimd(y[j],x);
				}
			}
		}
	}

	hkStringBuf sb; sb.printf("Special value test finished with crcs - Fp: 0x%x, FpDiv: 0x%x, Simd: 0x%x, SimdDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd(), opsTester.getCrcSimdDiv());
}

void testDenormsAndInfs(hkFpOpsTester& opsTester, hkPseudoRandomGenerator& randomGen)
{
	//force an addition/subtraction/multiply/divide into a denormalized outcome 
	{
		const float maxDenorm = *((float*)(&denormIntMax));
		const float maxNegDenorm = *((float*)(&negDenormIntMax));
		const float minDenorm = *((float*)(&denormIntMin));
		const float minNegDenorm = *((float*)(&negDenormIntMin));

		const uint32_t denormIntMaxInc = denormIntMax + 1;
		const uint32_t denormIntMaxDec = denormIntMax - 1;
		const uint32_t negDenormIntMinInc = negDenormIntMin + 1;
		const uint32_t negDenormIntMinDec = negDenormIntMin - 1;

		const float maxDenormInc = *((float*)(&denormIntMaxInc));
		const float maxDenormDec = *((float*)(&denormIntMaxDec));
		const float minNegDenormInc = *((float*)(&negDenormIntMinInc));
		const float minNegDenormDec = *((float*)(&negDenormIntMinDec));

		bool stopIt = false;
		for(uint32_t i = 0; i < NUM_SELECTIVE_DENORM_TESTS && !stopIt; ++i)
		{
			uint32_t xInt = randomGen.getRand32();
			{
				//selective addition/subtraction
				uint32_t cappedX = xInt & 0x8f8fffff; //only values with an exponent lower than 23 (roughly first five bits) can result in a denormalized float after add/sub
				float x = *((float*)(&cappedX));
				float subtractLower = x - maxDenorm; //setup the range which will produce a denormalized value when added/subtracted to/from x
				float subtractUpper = x - minNegDenorm;
				float subtractLowerDec = x - maxDenormInc;//calculate some extra boundary cases
				float subtractLowerInc = x - maxDenormDec;
				float subtractUpperDec = x - minNegDenormInc;
				float subtractUpperInc = x - minNegDenormDec;
				float y = (float)-randomGen.getRandRange( subtractLower, subtractUpper ); //choose a random value out of this range
				{
					opsTester.testAddFp(x, y);
					opsTester.testAddSimd(x, y);
					/*
					int result1Int = *((int*)(&result1));
					int result2Int = *((int*)(&result2));
					if( result1Int != result2Int )
					{
						stopIt = true;
						hkStringBuf sb; sb.printf("add arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: 0x%x \n", *((int*)(&x)), *((int*)(&y)), result1Int, result2Int);
					}
					*/
				}
				{
					opsTester.testSubFp(x, -y);
					opsTester.testSubSimd(x, -y);
					/*
					int result1Int = *((int*)(&result1));
					int result2Int = *((int*)(&result2));
					if( result1Int != result2Int )
					{
						stopIt = true;
						hkStringBuf sb; sb.printf("sub arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: -0x%x \n", *((int*)(&x)), *((int*)(&y)), result1Int, result2Int);
					}
					*/
				}
				//also test around denormal boundaries (including the area around 0)
				const float specialBounds[] = { x, x-minDenorm, x-maxNegDenorm, subtractLower, subtractUpper, subtractLowerDec, subtractLowerInc, subtractUpperDec, subtractUpperInc };
				for(int boundTest = 0; boundTest < sizeof(specialBounds)/sizeof(float); ++boundTest)
				{
					{
						opsTester.testAddFp(x, -specialBounds[boundTest]);
						opsTester.testAddSimd(x, -specialBounds[boundTest]);
						/*
						int result1Int = *((int*)(&result1));
						int result2Int = *((int*)(&result2));
						if( result1Int != result2Int )
						{
							stopIt = true;
							hkStringBuf sb; sb.printf("add arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: -0x%x \n", *((int*)(&x)), *((int*)(&specialBounds[boundTest])), result1Int, result2Int);
						}
						*/
					}
					{
						opsTester.testSubFp(x, specialBounds[boundTest]);
						opsTester.testSubSimd(x, specialBounds[boundTest]);
						/*
						int result1Int = *((int*)(&result1));
						int result2Int = *((int*)(&result2));
						if( result1Int != result2Int )
						{
							stopIt = true;
							hkStringBuf sb; sb.printf("sub arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: 0x%x \n", *((int*)(&x)), *((int*)(&specialBounds[boundTest])), result1Int, result2Int);
						}
						*/
					}
				}
			}
			{
				uint32_t yIntGen = randomGen.getRand32();

				//selective multiplication towards 0
				uint32_t cappedX = xInt & 0xbfffffff; //assuming normalized operands, only exponents of xExp < 127 (we take x <= 127 for convenience) can result in a denormalized float after multiplication...
				uint32_t xExp = (cappedX & 0x3f800000) >> 23;
				float x = *((float*)(&cappedX));
				float absX = x > 0 ? x : -x;
				{
					uint32_t yExp = 0x0000007F - xExp; //...specifically the float with an exponent of either 127 - xExp or 127 - (xExp + 1) (depending on the mantissa, we just calculate both and some more below) 
					uint32_t yInt = (yIntGen & 0x807fffff) | (yExp << 23); //mask away the random exponent and replace with the calcuted yExp
					float y = *((float*)(&yInt));
					float absY = y > 0 ? y : -y;
					{
						//for the chosen combination of x, y, we perform multiplication, but also for cases ( i:[1..p) | expandedX = x * 2^i, expandedY = y * 2^(-i)) where x > y
						float expandedX = absX > absY ? x : y;
						float expandedY = absX > absY ? y : x;
						for(uint32_t expandIntoDenormalizedStep = 0; expandIntoDenormalizedStep < 23; ++expandIntoDenormalizedStep, expandedX *= 2.0f, expandedY *= 0.5f)
						{
							opsTester.testMulFp(expandedX, expandedY);
							opsTester.testMulSimd(expandedX, expandedY);
							/*
							int result1Int = *((int*)(&result1));
							int result2Int = *((int*)(&result2));
							if( result1Int != result2Int )
							{
								stopIt = true;
								hkStringBuf sb; sb.printf("mul arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: 0x%x \n", *((int*)(&x)), *((int*)(&y)), result1Int, result2Int);
							}
							*/
						}
					}
				}
				//calculate some more until at least zero is reached
				for(uint32_t mantissaIndex = 1; mantissaIndex < 26; ++mantissaIndex) //25(not 23) is the max, as 127 - xExp may yield a result one exponent above denormal, and the result may be rounded upwards
				{
					uint32_t yExp = 0x0000007F - ((xExp+mantissaIndex) & 0x0000007F); //the second exponent test; in case of overflow, the test is useless, but perform it anyhow
					uint32_t yInt = (yIntGen & 0x807fffff) | (yExp << 23);
					float y = *((float*)(&yInt));
					float absY = y > 0 ? y : -y;
					{
						//for the chosen combination of x, y, we perform multiplication, but also for cases ( i:[1..p) | expandedX = x * 2^i, expandedY = y * 2^(-i)) where x > y
						float expandedX = absX > absY ? x : y;
						float expandedY = absX > absY ? y : x;
						for(uint32_t expandIntoDenormalizedStep = 0; expandIntoDenormalizedStep < 23; ++expandIntoDenormalizedStep, expandedX *= 0.5f, expandedY *= 0.5f)
						{
							opsTester.testMulFp(expandedX, expandedY);
							opsTester.testMulSimd(expandedX, expandedY);
							/*
							int result1Int = *((int*)(&result1));
							int result2Int = *((int*)(&result2));
							if( result1Int != result2Int )
							{
								stopIt = true;
								hkStringBuf sb; sb.printf("mul arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: 0x%x \n", *((int*)(&x)), *((int*)(&y)), result1Int, result2Int);
							}
							*/
						}
					}
				}

				//choose a mantissa for y that best approaches the boundary between denorm and norm floats, and take some samples around it for multiplication with x
				{
					uint32_t xNormalizedMantissaInt = (cappedX & 0x007fffff) | 0x3f800000;
					float xNormalizedMantissa = *((float*)(&xNormalizedMantissaInt));
					float yNormalizedMantissa = 2.0f / xNormalizedMantissa; //the boundary is defined by xMantissa * yMantissa = 2.0f
					uint32_t yMantissaBase = *((uint32_t*)(&yNormalizedMantissa));
					yMantissaBase -= 0x8;
					uint32_t yExp = 0x0000007F - xExp; 
					uint32_t yNoMantissa = (yIntGen & 0x80000000) | (yExp << 23);
					for(uint32_t mantissaStep = 0; mantissaStep < 0xF; ++mantissaStep, ++yMantissaBase)
					{
						uint32_t yInt = yNoMantissa | (yMantissaBase & 0x007fffff); 
						float y = *((float*)(&yInt));
						opsTester.testMulFp(x, y);
						opsTester.testMulSimd(x, y);
						/*
						int result1Int = *((int*)(&result1));
						int result2Int = *((int*)(&result2));
						if( result1Int != result2Int )
						{
							stopIt = true;
							hkStringBuf sb; sb.printf("mul arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: 0x%x \n", *((int*)(&x)), *((int*)(&y)), result1Int, result2Int);
							uint32_t controlWordfp, controlWordSse;
							__control87_2(_PC_53, _MCW_PC, &controlWordfp, &controlWordSse);
							DoubleUnion xDouble;
							DoubleUnion yDouble;
							DoubleUnion resultDouble;
							xDouble.m_float64 = x;
							yDouble.m_float64 = y;
							hkStringBuf sb; sb.printf("arg1part1: 0x%x, arg1part2: 0x%x, arg2part1: 0x%x, arg2part2: 0x%x \n", xDouble.m_int32[0], xDouble.m_int32[1], yDouble.m_int32[0], yDouble.m_int32[1]);
							resultDouble.m_float64 = opsTester.testMulDouble(xDouble.m_float64,yDouble.m_float64);
							hkStringBuf sb; sb.printf("resultpart1: 0x%x, resultpart2: 0x%x \n", resultDouble.m_int32[0], resultDouble.m_int32[1]);
							opsTester.testMulFp(x, y);
							opsTester.testMulSimd(x, y);
						}
						*/
					}
				}

				//selective division towards 0
				for(uint32_t mantissaIndex = 0; mantissaIndex < 26; ++mantissaIndex)
				{
					uint32_t yExp = 0x0000007E + xExp + mantissaIndex; //ignore overflow, test these cases anyhow
					uint32_t yInt = (yIntGen & 0x807fffff) | (yExp << 23);
					float y = *((float*)(&yInt));
					{
						opsTester.testDivFp(x, y);
						opsTester.testDivSimd(x, y);
						/*
						int result1Int = *((int*)(&result1));
						int result2Int = *((int*)(&result2));
						if( result1Int != result2Int )
						{
							stopIt = true;
							hkStringBuf sb; sb.printf("div arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: 0x%x \n", *((int*)(&x)), *((int*)(&y)), result1Int, result2Int);
						}
						*/
					}
				}

				//choose a mantissa for y that best approaches the boundary between denorm and norm floats, and take some samples around it for division with x
				{
					uint32_t yMantissaBase = cappedX & 0x007fffff;
					yMantissaBase -= 0x8;
					uint32_t yExp = 0x0000007E + xExp; //from non-denorm (y < x) to denorm (y > x)
					uint32_t yNoMantissa = (yIntGen & 0x80000000) | (yExp << 23);
					for(uint32_t mantissaStep = 0; mantissaStep < 0xF; ++mantissaStep, ++yMantissaBase)
					{
						uint32_t yInt = yNoMantissa | (yMantissaBase & 0x007fffff); 
						float y = *((float*)(&yInt));
						opsTester.testDivFp(x, y);
						opsTester.testDivSimd(x, y);
						/*
						int result1Int = *((int*)(&result1));
						int result2Int = *((int*)(&result2));
						if( result1Int != result2Int )
						{
							stopIt = true;
							hkStringBuf sb; sb.printf("div arg1: 0x%x, arg2: 0x%x, Result1: 0x%x, Result2: 0x%x \n", *((int*)(&x)), *((int*)(&y)), result1Int, result2Int);
						}
						*/
					}
				}
			}
		
		}
	}

	//test operations for random denormalized sources
	for(uint32_t i = 0; i < NUM_DENORM_TESTS; ++i)
	{
		uint32_t xInt = randomGen.getRand32();
		uint32_t yInt = randomGen.getRand32();
		xInt = xInt & 0x807fffff;
		yInt = yInt & 0x807fffff;
		float x = *((float*)(&xInt));
		float y = *((float*)(&yInt));

		{
			opsTester.testAddFp(x, y);
			opsTester.testAddFp(y, x);
			opsTester.testAddSimd(x, y);
			opsTester.testAddSimd(y, x);
		}
		{
			opsTester.testSubFp(x, y);
			opsTester.testSubFp(y, x);
			opsTester.testSubSimd(x, y);
			opsTester.testSubSimd(y, x);
		}
		{
			opsTester.testMulFp(x, y);
			opsTester.testMulFp(y, x);
			opsTester.testMulSimd(x, y);
			opsTester.testMulSimd(y, x);
		}
		{
			opsTester.testDivFp(x, y);
			opsTester.testDivFp(y, x);
			opsTester.testDivSimd(x, y);
			opsTester.testDivSimd(y, x);
		}
	}

	for(uint32_t i = 0; i < NUM_DENORM_NORM_TESTS; ++i)
	{
		uint32_t xInt = randomGen.getRand32();
		uint32_t yInt = randomGen.getRand32();
		xInt = xInt & 0x807fffff;
		uint32_t ySign = (yInt & 0x80000000);
		yInt = (yInt & 0x7fffffff) > 0x7f7fffff ? (ySign | 0x7f7fffff) : yInt; //yInt is never larger than the largest normalized float
		float x = *((float*)(&xInt));
		float y = *((float*)(&yInt));

		{
			opsTester.testAddFp(x, y);
			opsTester.testAddFp(y, x);
			opsTester.testAddSimd(x, y);
			opsTester.testAddSimd(y, x);
		}
		{
			opsTester.testSubFp(x, y);
			opsTester.testSubFp(y, x);
			opsTester.testSubSimd(x, y);
			opsTester.testSubSimd(y, x);
		}
		{
			opsTester.testMulFp(x, y);
			opsTester.testMulFp(y, x);
			opsTester.testMulSimd(x, y);
			opsTester.testMulSimd(y, x);
		}
		{
			opsTester.testDivFp(x, y);
			opsTester.testDivFp(y, x);
			opsTester.testDivSimd(x, y);
			opsTester.testDivSimd(y, x);
		}
	}

	//force an addition/subtraction/multiply/divide into a infinite outcome (similar to the denormalized cases)
	for(uint32_t i = 0; i < NUM_SELECTIVE_INF_TESTS; ++i)
	{
		uint32_t xInt = randomGen.getRand32();

		//addition and subtraction
		{
			const uint32_t sign = (xInt & 0x80000000);
			const uint32_t normInfBorderInt = sign | 0x7f7fffff; //last valid normalized float
			const uint32_t maxNormFromBorderInt = sign | 0x73000000; //24 bits down
			const float normInfBorder = *((float*)(&normInfBorderInt));
			const float maxNormFromBorder = *((float*)(&maxNormFromBorderInt));
			
			float x = (float)randomGen.getRandRange(normInfBorder, maxNormFromBorder);

			float minDiff = normInfBorder - x;
			float maxDiff = normInfBorder;

			float y = (float)randomGen.getRandRange(minDiff, maxDiff);
			{
				opsTester.testAddFp(x, y);
				opsTester.testAddSimd(x, y);
			}
			{
				opsTester.testSubFp(x, -y);
				opsTester.testSubSimd(x, -y);
			}

			uint32_t minDiffInt = *((uint32_t*)(&minDiff));
			uint32_t baseY = (minDiffInt & 0x7fffffff) > 8 ? minDiffInt - 8 : sign;  //absolute is larger than 8, subtract 8, otherwise simply set to zero (sign bit stays the same)
			for( int j = 0; j < 0xFF; ++j, ++baseY )
			{
				y = *((float*)(&baseY));
				{
					opsTester.testAddFp(x, y);
					opsTester.testAddSimd(x, y);
				}
				{
					opsTester.testSubFp(x, -y);
					opsTester.testSubSimd(x, -y);
				}	
			}
			
		}
		//multiplication and division
		{
			uint32_t cappedX = xInt | 40000000; //only exponents of 128 (2^1) or higher can have a multiplication with a term different from inf that has inf as result, so force this
			float x = *((float*)(&cappedX));
			uint32_t xExp = (cappedX & 0x7f800000) >> 23;
			uint32_t yExpOffsetFromZero = (0x7f - xExp & 0x7f);
			uint32_t yExpMul = yExpOffsetFromZero + 0x7f; //the minimum exponent to get an inf result for mul
			uint32_t yExpDiv = 0x7f - yExpOffsetFromZero; //the maximum exponent to get an inf result for div

			{
				//test the boundary case
				uint32_t yInt = (yExpMul << 23);
				float y = *((float*)(&yInt));
				opsTester.testMulFp(x, y);
				opsTester.testMulSimd(x, y);
			}
			{
				//test division as well
				uint32_t yInt = (((yExpDiv+1) & 0x7f) << 23);
				float y = *((float*)(&yInt));
				opsTester.testDivFp(x, y);
				opsTester.testDivSimd(x, y);
			}

			//generate some more y values that result in an infinite outcome
			for( int j = 0; j < 0xff; ++j)
			{
				uint32_t yIntGen = randomGen.getRand32();
				uint32_t yIntGenExp = (yIntGen & 0x7f800000) >> 23;
				uint32_t yExpMax = yExpMul > yIntGenExp ? yExpMul : yIntGenExp;
				uint32_t yExpMin = yExpDiv > yIntGenExp ? yIntGenExp : yExpDiv;
				{
					uint32_t yInt = (yIntGen & 0x807fffff) | (yExpMax << 23);
					float y = *((float*)(&yInt));
					opsTester.testMulFp(x, y);
					opsTester.testMulSimd(x, y);
				}
				{
					uint32_t yInt = (yIntGen & 0x807fffff) | (yExpMin << 23);
					float y = *((float*)(&yInt));
					opsTester.testDivFp(x, y);
					opsTester.testDivSimd(x, y);
				}
			}

			//probe the inf borders, both for mul and div
			{
				uint32_t yIntGen = randomGen.getRand32();
				//choose a mantissa for y that best approaches the boundary between norma and inf floats, and take some samples around it for multiplication with x
				{
					uint32_t xNormalizedMantissaInt = (cappedX & 0x007fffff) | 0x3f800000;
					float xNormalizedMantissa = *((float*)(&xNormalizedMantissaInt));
					float yNormalizedMantissa = 2.0f / xNormalizedMantissa; //the boundary is defined by xMantissa * yMantissa = 2.0f
					uint32_t yMantissaBase = *((uint32_t*)(&yNormalizedMantissa));
					yMantissaBase -= 0x8;
					uint32_t yNoMantissa = (yIntGen & 0x80000000) | ((yExpMul-1) << 23);
					for(uint32_t mantissaStep = 0; mantissaStep < 0xF; ++mantissaStep, ++yMantissaBase)
					{
						uint32_t yInt = yNoMantissa | (yMantissaBase & 0x007fffff); 
						float y = *((float*)(&yInt));
						opsTester.testMulFp(x, y);
						opsTester.testMulSimd(x, y);
					}
				}

				//choose a mantissa for y that best approaches the boundary between denorm and norm floats, and take some samples around it for division with x
				{
					uint32_t yMantissaBase = cappedX & 0x007fffff;
					yMantissaBase -= 0x8;
					uint32_t yNoMantissa = (yIntGen & 0x80000000) | (yExpDiv << 23);
					for(uint32_t mantissaStep = 0; mantissaStep < 0xF; ++mantissaStep, ++yMantissaBase)
					{
						uint32_t yInt = yNoMantissa | (yMantissaBase & 0x007fffff); 
						float y = *((float*)(&yInt));
						opsTester.testDivFp(x, y);
						opsTester.testDivSimd(x, y);
					}
				}
			}
		}

	}

	hkStringBuf sb; sb.printf("Denorms and infs test finished with crcs - Fp: 0x%x, FpDiv: 0x%x, Simd: 0x%x, SimdDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd(), opsTester.getCrcSimdDiv());
}

void testRandomNumbers(hkFpOpsTester& opsTester, hkPseudoRandomGenerator& randomGen)
{
	//just spawn some random numbers and test all ops
	for(uint32_t i = 0; i < NUM_RANDOM_TESTS; ++i)
	{
		float x = getRandRealFullRange(randomGen);
		float y = getRandRealFullRange(randomGen);
		{
			opsTester.testAddFp(x, y);
			opsTester.testAddSimd(x, y);
		}
		{
			opsTester.testSubFp(x, y);
			opsTester.testSubSimd(x, y);
		}
		{
			opsTester.testMulFp(x, y);
			opsTester.testMulSimd(x, y);
		}
		{
			opsTester.testDivFp(x, y);
			opsTester.testDivSimd(x, y);
		}
	}

	hkStringBuf sb; sb.printf("Random numbers test finished with crcs - Fp: 0x%x, FpDiv: 0x%x, Simd: 0x%x, SimdDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd(), opsTester.getCrcSimdDiv());
}

void testRandomTranscendental(hkFpOpsTester& opsTester, hkPseudoRandomGenerator& randomGen)
{
	uint32_t numTests = 0;

	//just spawn some random numbers and test all ops
	for(; numTests < NUM_RANDOM_TRANSCENDENTAL_TESTS; ++numTests)
	{
		float x = *((float*)(&numTests));//randomGen.getRandRange(-pi,pi);
		{
			opsTester.testSqrtFp(x);
		}
		{
			opsTester.testSinFp(x);
		}
		{
			opsTester.testCosFp(x);
		}
		{
			opsTester.testTanFp(x);
		}
		{
			opsTester.testLogFp(x);
		}
		{
			opsTester.testSqrtSimd(x);
		}
	}
	{
		float x = *((float*)(&numTests));
		{
			opsTester.testSqrtFp(x);
		}
		{
			opsTester.testSinFp(x);
		}
		{
			opsTester.testCosFp(x);
		}
		{
			opsTester.testTanFp(x);
		}
		{
			opsTester.testLogFp(x);
		}
		{
			opsTester.testSqrtSimd(x);
		}
	}

	for(uint32_t i = 0; i < NUM_RANDOM_TESTS; ++i)
	{
		float x = getRandRealFullRange(randomGen);
		float y = getRandRealFullRange(randomGen);
		opsTester.testAtan2Fp(x, y);
	}

	hkStringBuf sb; sb.printf("Random input transcendental tests finished with crcs - Fp: 0x%x, FpDiv: 0x%x, Simd:0x%x\n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd());
}

void testContraction(hkFpOpsTester& opsTester)
{
	{
		float x = 1.0f;
		float y = 2.0f;
		float z = 3.0f;

		opsTester.testContractionFp(x,y,z);
		opsTester.testContractionSimd(x,y,z);
	}
	hkStringBuf sb; sb.printf("Contraction test finished with crcs - Fp: 0x%x, Simd: 0x%x \n", opsTester.getCrcFpContract(), opsTester.getCrcSimdContract());
}

void printAllResults(hkFpOpsTester& opsTester)
{
	hkStringBuf sb; sb.printf("Crc check - Fp: 0x%x, FpDiv: 0x%x, Simd: 0x%x, SimdDiv: 0x%x \n", opsTester.getCrcFp(), opsTester.getCrcFpDiv(), opsTester.getCrcSimd(), opsTester.getCrcSimdDiv());

#ifndef FULL_ACC_RESULTS 
#ifdef FP_TEST_SMALL
	HK_TEST( opsTester.getCrcFp() == 0xc31e6212 && opsTester.getCrcFpDiv() == 0x6ed31af8 && opsTester.getCrcSimd() == 0x190bd4ef && opsTester.getCrcSimdDiv() == 0x393c0817 );//24
#else
	HK_TEST( opsTester.getCrcFp() == 0x1a61662a && opsTester.getCrcFpDiv() == 0x532d5617 && opsTester.getCrcSimd() == 0x5c49597d && opsTester.getCrcSimdDiv() == 0x82a443a5 );//24	
#endif
#else
#ifdef FP_TEST_SMALL
	HK_TEST( opsTester.getCrcFp() == 0xc31e6212 && opsTester.getCrcFpDiv() == 0x6ed31af8 && opsTester.getCrcSimd() == 0x83136b6c && opsTester.getCrcSimdDiv() == 0x3d7d94ec );//24
#else
	HK_TEST( opsTester.getCrcFp() == 0x1a61662a && opsTester.getCrcFpDiv() == 0x532d5617 && opsTester.getCrcSimd() == 0xef32b457 && opsTester.getCrcSimdDiv() == 0x261bf9bf );//24	
#endif
#endif
}

int FpTest_main()
{
	if( !testSse2() )
	{
		hkStringBuf sb; sb.printf("Aborting tests. \n");
		return 0;
	}

	hkFpOpsTester opsTester;
	hkPseudoRandomGenerator pseudoRandomGenerator(180673);

	{
		hkStringBuf sb; sb.printf("Fp consistency test. \n---------------------------\n");
	}

	showCpuInfo();

#ifndef _WIN64
	uint32_t controlWordfp, controlWordSse, oldControlWordfp, oldControlWordSse;

	__control87_2(0, 0, &oldControlWordfp, &oldControlWordSse);
	__control87_2(_PC_64, _MCW_PC, &controlWordfp, &controlWordSse);
	__control87_2(_RC_NEAR, _MCW_RC, &controlWordfp, &controlWordSse);
	__control87_2(_MCW_DN, _DN_SAVE, &controlWordfp, &controlWordSse);
#endif

	//testRegWidthDoubleExtended(opsTester);

#ifndef _WIN64
	__control87_2(_PC_53, _MCW_PC, &controlWordfp, &controlWordSse);
#endif

	//testOpWidth(opsTester,pseudoRandomGenerator);
	testRegWidthDouble(opsTester,pseudoRandomGenerator);

#ifndef _WIN64
	__control87_2(_PC_24, _MCW_PC, &controlWordfp, &controlWordSse);

	{
		hkStringBuf sb; sb.printf("Controlwordfp: 0x%x, Controlworldsse: 0x%x \n\n", controlWordfp, controlWordSse);
	}
#endif


	//showRegWidth(opsTester);

	testRegWidth(opsTester,pseudoRandomGenerator);
	testRoundingModeAndBoundaries(opsTester,pseudoRandomGenerator);
	testSpecialValues(opsTester,pseudoRandomGenerator);
	testDenormsAndInfs(opsTester,pseudoRandomGenerator);
	testRandomNumbers(opsTester,pseudoRandomGenerator);
	testRandomTranscendental(opsTester,pseudoRandomGenerator);

	//testContraction(opsTester);

	printAllResults(opsTester);
	
#ifndef _WIN64
	//restore fp settings
	__control87_2(oldControlWordfp, 0xFFFFFFFF, &controlWordfp, 0);
	__control87_2(oldControlWordSse, 0xFFFFFFFF, 0, &controlWordSse);
	HK_ASSERT(0x1923ACDF, oldControlWordfp == controlWordfp);
	HK_ASSERT(0x1923ACDF, oldControlWordSse == controlWordSse);
#endif

	return 0;
}


#else // !defined( HK_PLATFORM_WIN32 ) && !defined( HK_PLATFORM_X64 )


int FpTest_main()
{
	return 0;
}


#endif


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
//HK_TEST_REGISTER(FpTest_main, "Fast", "Common/Test/UnitTest/Internal/", __FILE__);

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

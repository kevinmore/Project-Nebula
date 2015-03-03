/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Internal/hknpInternal.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>

#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>

#define HK_PERF_REPORT(NAME, VALUE)	HK_MULTILINE_MACRO_BEGIN															\
	char msgBuf[512];								\
	hkOstream msg(msgBuf, sizeof(msgBuf), true);	\
	msg << "[PERF:" << NAME << ":" << VALUE << "]";			\
	hkTestReportFunction( false, msgBuf, __FILE__, __LINE__); \
	HK_MULTILINE_MACRO_END

int MxVectorTests_main()
{
// 	hkMxVector<4> v(hkSimdReal::getConstant(HK_QUADREAL_1));
// 	HK_TEST( v.isOk4() );
//
// 	hkStopwatch timer;
//
// 	hknpSolverInfo solverInfo;
// 	int numSolverVel = 32 * 1024;
//
// 	hkArray<hknpSolverVelocity> solverVels; solverVels.reserve(numSolverVel);
// 	hkArray<hknpSolverSumVelocity> sumVels; sumVels.reserve(numSolverVel);
// 	{
// 		for (int i=0; i<numSolverVel; ++i)
// 		{
// 			hknpSolverVelocity sV; sV.setZero();
// 			solverVels.pushBackUnchecked(sV);
//
// 			hknpSolverSumVelocity ssV; ssV.setZero();
// 			sumVels.pushBackUnchecked(ssV);
// 		}
// 	}
//
// 	hkArray<hknpMotionProperties> motionProperties;
// 	{
// 		hknpMotionProperties mp( hknpMotionPropertiesId::DYNAMIC );
// 		motionProperties.pushBack(mp);
// 	}
//
// 	const int numRepeat = 1000;
// 	{
// 		timer.reset();
// 		timer.start();
// 		for (int s=0; s<numRepeat; ++s)
// 			hknpMotionUtil::_subIntegrateSolverVelocities<1>(solverVels.begin(), sumVels.begin(), numSolverVel, solverInfo, motionProperties.begin());
// 		timer.stop();
//
// 		HK_PERF_REPORT("_subIntegrateAllMotions<1>", timer.getElapsedSeconds());
// 	}
// 	{
// 		timer.reset();
// 		timer.start();
// 		for (int s=0; s<numRepeat; ++s)
// 			hknpMotionUtil::_subIntegrateSolverVelocities<2>(solverVels.begin(), sumVels.begin(), numSolverVel, solverInfo, motionProperties.begin());
// 		timer.stop();
//
// 		HK_PERF_REPORT("_subIntegrateAllMotions<2>", timer.getElapsedSeconds());
// 	}
// 	{
// 		timer.reset();
// 		timer.start();
// 		for (int s=0; s<numRepeat; ++s)
// 			hknpMotionUtil::_subIntegrateSolverVelocities<4>(solverVels.begin(), sumVels.begin(), numSolverVel, solverInfo, motionProperties.begin());
// 		timer.stop();
//
// 		HK_PERF_REPORT("_subIntegrateAllMotions<4>", timer.getElapsedSeconds());
// 	}
// 	{
// 		timer.reset();
// 		timer.start();
// 		for (int s=0; s<numRepeat; ++s)
// 		hknpMotionUtil::_subIntegrateAllMotions<8>(solverVels.begin(), sumVels.begin(), numSolverVel, solverInfo, motionProperties.begin());
// 		timer.stop();
//
// 		HK_PERF_REPORT("_subIntegrateAllMotions<8>", timer.getElapsedSeconds());
// 	}
// 	{
// 		timer.reset();
// 		timer.start();
// 		for (int s=0; s<numRepeat; ++s)
// 			hknpMotionUtil::_subIntegrateAllMotions<16>(solverVels.begin(), sumVels.begin(), numSolverVel, solverInfo, motionProperties.begin());
// 		timer.stop();
//
// 		HK_PERF_REPORT("_subIntegrateAllMotions<16>", timer.getElapsedSeconds());
// 	}
// 	{
// 		timer.reset();
// 		timer.start();
// 		for (int s=0; s<numRepeat; ++s)
// 			hknpMotionUtil::_subIntegrateAllMotions<32>(solverVels.begin(), sumVels.begin(), numSolverVel, solverInfo, motionProperties.begin());
// 		timer.stop();
//
// 		HK_PERF_REPORT("_subIntegrateAllMotions<32>", timer.getElapsedSeconds());
// 	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

// os.todo  convert this into a performance test
HK_TEST_REGISTER(MxVectorTests_main, "Perf NP", "Physics/Test/UnitTest/Internal/", __FILE__ );

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

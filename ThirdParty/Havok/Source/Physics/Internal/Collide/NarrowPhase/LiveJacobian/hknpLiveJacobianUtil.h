/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_LIVE_JACOBIAN_UTIL_H
#define HKNP_LIVE_JACOBIAN_UTIL_H

#include <Physics/Physics/Collide/NarrowPhase/hknpCollidePipeline.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>
#include <Physics/Physics/Collide/NarrowPhase/LiveJacobian/hknpLiveJacobianInfo.h>

class hknpSimulationThreadContext;
struct hknpLiveJacobianInfo;
class hknpSimulationContext;
class hknpConstraintAtomJacobianStream;
struct hknpConstraintSolverJacobianRange2;
struct hkcdManifold4;


/// Solver utility functions.
namespace hknpLiveJacobianUtil
{
	/// Helper to generate live Jacobians by integrating the passed in velocities and rebuilding contact Jacobians.
	void HK_CALL generateLiveJacobians(
		const hknpSimulationThreadContext& tl, const hknpSolverStep& solverStep,
		const hknpSolverSumVelocity* solverSumVelAStream, const hknpSolverSumVelocity* solverSumVelBStream,
		const hknpLiveJacobianInfoRange* liveJacInfos );
}


#endif // HKNP_LIVE_JACOBIAN_UTIL_H

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

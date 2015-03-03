/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_SOLVER_INTEGRATOR_H
#define HKNP_SOLVER_INTEGRATOR_H

#include <Common/Base/Math/Vector/Mx/hkMxVector.h>
#include <Common/Base/Types/Geometry/IntSpaceUtil/hkIntSpaceUtil.h>

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverVelocity.h>

class hkBlockStreamCommandWriter;
class hknpBody;
class hknpMotion;
class hknpMotionProperties;
struct hknpSpaceSplitterData;
struct hknpSolverStep;


/// Internal helper class to deal with motions and swepted aabbs of bodies.
class hknpSolverIntegrator
{
	public:

		/// Integrate solver velocities.
		static void HK_CALL subIntegrate(
			const hknpSimulationThreadContext& tl, const hknpSolverStep& solverStep,
			hknpSolverVelocity* HK_RESTRICT solverVels, hknpSolverSumVelocity* HK_RESTRICT sumVels, int numSolverVel,
			const hknpSolverInfo& solverInfo, const hknpMotionProperties* HK_RESTRICT motionProperties );

		template <int MXLENGTH>
		HK_FORCE_INLINE static void HK_CALL _stabilize(
			const hknpSolverInfo& solverInfo, const hknpMotionProperties* HK_RESTRICT motionProperties[],
			hkMxVector<MXLENGTH>& velInOut );

		/// Integrate solver velocities, MX version.
		template <int MXLENGTH>
		HK_FORCE_INLINE static void HK_CALL _subIntegrate(
			const hknpSimulationThreadContext& tl, const hknpSolverStep& solverStep,
			hknpSolverVelocity* HK_RESTRICT solverVels, hknpSolverSumVelocity* HK_RESTRICT sumVels, int numSolverVel,
			const hknpSolverInfo& solverInfo, const hknpMotionProperties* HK_RESTRICT motionProperties );

		/// Integrate solver velocities after the last solver iteration.
		static void HK_CALL subIntegrateLastStep(
			const hknpSolverStep& solverStep,
			hknpSolverVelocity* solverVels, hknpSolverSumVelocity* sumVels, int numSolverVel,
			hknpMotion* motions, hknpDeactivationState* deactivationState,
			const hknpMotionProperties* HK_RESTRICT motionProperties,
			class hknpSolverStepInfo* solverManager, const hknpSimulationThreadContext& tl,
			struct hknpSpaceSplitterData* spaceSplitterData,
			hkUint32* solverVelOkToDeactivateBits, int solverVelOkToDeactivateBitOffset, int bitCapacity );

		/// Integrate solver velocities after the last solver iteration, MX version.
		template <int MXLENGTH>
		HK_FORCE_INLINE static void HK_CALL _subIntegrateLastStep(
			const hknpSolverStep& solverStep,
			hknpSolverVelocity* solverVels, hknpSolverSumVelocity* sumVels, int numSolverVel,
			hknpMotion* motions, hknpDeactivationState* deactivationState,
			const hknpMotionProperties* HK_RESTRICT motionProperties,
			class hknpSolverStepInfo* solverManager, const hknpSimulationThreadContext& tl,
			struct hknpSpaceSplitterData* spaceSplitterData,
			hkUint32* solverVelOkToDeactivateBits, int solverVelOkToDeactivateBitOffset, int bitCapacity );

		/// Integrate motions after the last solver step.
		static void HK_CALL integrateMotions(
			const hknpSolverStep& solverStep,
			hknpSolverVelocity* HK_RESTRICT solverVels, hknpSolverSumVelocity* HK_RESTRICT sumVels,
			hknpMotion* HK_RESTRICT motions, hknpDeactivationState* HK_RESTRICT deactivationState, const int* motionIds,
			const hknpMotionProperties* motionProperties,
			class hknpSolverStepInfo* solverManager, const hknpSimulationThreadContext& tl,
			struct hknpSpaceSplitterData* spaceSplitterData, hkUint32 deactivate4Bits[] );

		/// Integrate motions after the last solver step (MX version).
		template <int MXLENGTH>
		HK_FORCE_INLINE static void HK_CALL _integrateMotions(
			const hknpSolverStep& solverStep,
			hknpSolverVelocity* HK_RESTRICT solverVels, hknpSolverSumVelocity* HK_RESTRICT sumVels,
			hknpMotion* HK_RESTRICT motions, hknpDeactivationState* HK_RESTRICT deactivationState,
			const int* HK_RESTRICT motionIdsIn, const hknpMotionProperties* HK_RESTRICT motionProperties,
			hknpSolverStepInfo* HK_RESTRICT solverManager, const hknpSimulationThreadContext& tl,
			struct hknpSpaceSplitterData* spaceSplitterData, hkUint32 deactivate4Bits[] );

		template <int MXLENGTH>
		HK_FORCE_INLINE static void _fixupNansAndClipVelocities( const hknpSolverInfo &solverInfo,
			const hknpMotionProperties* HK_RESTRICT motionProperties[], hkMxVector<MXLENGTH> &linVel,
			hkMxVector<MXLENGTH> &angVel );
};


#endif // HKNP_SOLVER_VEL_UTIL_H

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

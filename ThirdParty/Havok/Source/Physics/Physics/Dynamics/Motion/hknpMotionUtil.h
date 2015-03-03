/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_MOTION_UTIL_H
#define HKNP_MOTION_UTIL_H

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Base/Types/Geometry/IntSpaceUtil/hkIntSpaceUtil.h>

#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverVelocity.h>

class hknpBody;
class hknpMotionProperties;
class hknpSimulationThreadContext;


/// Internal helper class to deal with motions and swept AABBs of bodies.
class hknpMotionUtil
{
	public:

		//
		// TIM utilities
		//

		/// Convert angle to an angular TIM.
		static HK_FORCE_INLINE int convertAngleToAngularTIM( hkReal angle );

		/// Convert angle to an angular TIM.
		static HK_FORCE_INLINE void convertAngleToAngularTIM( hkSimdRealParameter angle, hkInt32* angleOut );

		/// Calculate the angle between two vectors in [0..255]  (PI/2)
		static HK_FORCE_INLINE int convertAngleToAngularTIM( hkVector4Parameter planeDirA, hkVector4Parameter planeDirB );

		/// Convert distance to linear TIM.
		static HK_FORCE_INLINE void convertDistanceToLinearTIM(
			const hknpSolverInfo& solverInfo, hkSimdRealParameter deltaDist, hkUint16& linearTimOut );

		/// Convert linear velocity to linear TIM.
		static HK_FORCE_INLINE void convertVelocityToLinearTIM(
			const hknpSolverInfo& solverInfo, hkVector4Parameter velocity, hkUint16& linearTimOut );

		//
		// AABB utilities
		//

		/// Calculate the AABB of a static body.
		static HK_FORCE_INLINE void HK_CALL calcStaticBodyAabb(
			const hknpBody& body, hkSimdRealParameter collisionTolance, hkAabb* HK_RESTRICT aabbOut );

		/// Internal function which calculates how to sweep an AABB.
		static void HK_CALL calcSweepExpansion(
			const hknpBody* HK_RESTRICT body, const hknpMotion& motion,
			hkSimdRealParameter collisionTolerance, const hknpSolverInfo& solverInfo,
			hkVector4* linExpansion0, hkVector4* HK_RESTRICT linExpansion1,
			hkSimdReal* anyDirectionExpansionOut, hkSimdReal* linearDirectionExpansionOut );

		/// Expand an AABB so that it includes the future transform of the body.
		/// The body shape is passed as a separate parameter to allow SPU to keep the PPU shape address in the body.
		static void HK_CALL calcFutureAabb(
			const hknpBody& body, const hknpShape* shape, const hknpMotion& motion, hkSimdRealParameter deltaTime,
			hkAabb* aabbOut );

		/// Internal function which sweeps an AABB and updates space-time based collision data in the body.
		/// The body shape is passed as a separate parameter to allow SPU to keep the PPU shape address in the body.
		static HK_FORCE_INLINE void _sweepBodyAndCalcAabb(
			hknpBody* body, const hknpShape* shape, const hknpMotion& motion, const hknpBodyQuality& quality,
			hkSimdRealParameter collisionTolerance, const hknpSolverInfo& solverInfo, hkAabb* aabbOut );

		/// Calculate the swept AABB of a body.
		static void HK_CALL calcSweptBodyAabb(
			hknpBody* body, const hknpMotion& motion, const hknpBodyQuality& quality,
			hkSimdRealParameter collisionTolerance, const hknpSolverInfo& solverInfo, hkIntSpaceUtil& intspaceUtil );

		//
		// Integration utilities
		//

		/// Estimate a motion's COM and orientation at a relative time, based on current velocity.
		static HK_FORCE_INLINE void HK_CALL _predictMotionTransform(
			hkSimdRealParameter deltaTime, const hknpMotion& motion,
			hkVector4* HK_RESTRICT motionComOut, hkQuaternion* HK_RESTRICT motionOrientationOut );

		/// Calculate a body's world space transform given a COM and orientation for its motion.
		static HK_FORCE_INLINE void HK_CALL _calculateBodyTransform(
			const hknpBody& body, hkVector4Parameter motionCom, hkQuaternionParameter motionOrientation,
			hkTransform* HK_RESTRICT worldFromBodyOut );

		/// Build solver velocities from motions, allocating into the output arrays.
		static void HK_CALL buildSolverVelocities(
			hknpSimulationThreadContext* tl,
			hknpWorld* world, const hkArray<hknpMotionId> &solverIdToMotionId,
			hkArray<hknpSolverVelocity>& solverVelocities, hkArray<hknpSolverSumVelocity>& solverSumVelocities );

		/// Build solver velocities from motions, using preallocated output buffers.
		static void HK_CALL gatherSolverVelocities(
			hknpSimulationThreadContext* tl,
			hknpWorld* world, const hknpMotionId* motionIds, int numIds,
			hknpSolverVelocity* HK_RESTRICT solverVelsOut, hknpSolverSumVelocity* HK_RESTRICT solverSumVelsOut );

		/// Integrate a motion, given linear (in world space) and angular (in local space) velocities.
		static void HK_CALL integrateMotionTransform(
			hkVector4Parameter linVel, hkVector4Parameter angVelLocal,
			hkSimdRealParameter deltaTime, hknpMotion* HK_RESTRICT motionsInOut );

		/// Update a body after its motion has been integrated.
		static void HK_CALL updateBodyAfterMotionIntegration( hknpWorld* world, hknpBodyId bodyId );

		/// Update all relevant bodies after a set of motions has been integrated.
		static void HK_CALL updateAllBodies(
			const hknpSimulationThreadContext* tl,
			const hknpBodyId* dynamicBodyIds, int numDynamicBodyIds,
			hknpBody* HK_RESTRICT allBodies, hknpMotion* HK_RESTRICT allMotions,
			const hknpBodyQuality* HK_RESTRICT allQualities,
			const hknpSolverInfo& solverInfo, const hkIntSpaceUtil& intSpaceUtil );

		/// Once a motion's center of mass has changed,
		/// this will update the cellIndex in the motion and inform the motion manager.
		static void HK_CALL updateCellIdx( hknpWorld* world, hknpMotion* motion, hknpMotionId motionId );

		//
		// Other
		//

		/// Calculates the energy of a motion as the sum of kinetic energy (linear and angular) plus
		/// gravitational potential energy relative to a ground plane.
		static HK_FORCE_INLINE hkSimdReal calcEnergy(
			const hknpMotion& motion, hkVector4Parameter gravity, hkSimdRealParameter groundHeight );
};

#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.inl>


#endif // HKNP_MOTION_UTIL_H

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

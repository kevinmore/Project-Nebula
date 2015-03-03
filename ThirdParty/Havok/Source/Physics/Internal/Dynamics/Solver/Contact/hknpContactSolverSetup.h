/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONTACT_SOLVER_SETUP_H
#define HKNP_CONTACT_SOLVER_SETUP_H

#include <Physics/Physics/hknpTypes.h>

#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


class hknpSolverSumVelocity;

/// Setup of contact constraints.
class hknpContactSolverSetup
{
	public:

		enum BuildJacConfig
		{
			BUILD_USING_CACHE,
			BUILD_NO_CACHE,
			REBUILD_WITH_CACHE,
			REBUILD_LINEAR_ONLY_WITH_CACHE,
		};

		struct BuildContactInput
		{
			BuildContactInput() { m_friction = 0.5f; m_maxImpulse = 0.5f; m_fractionOfClippedImpulseToApply = 1.0f; }
			hkReal m_friction;
			hkReal m_maxImpulse;
			hkReal m_fractionOfClippedImpulseToApply;
		};

	public:

		static void HK_CALL initConstants( hkReal linearErrorRecoveryVelocity = 1.0f, hkReal exponentialErrorRecoveryVelocity = 0.05f );

		struct BuildConfig
		{
			void init( const hknpMotion* motionA, const hknpMotion* motionB );
			HK_FORCE_INLINE void _init( const hknpMotion* motionA, const hknpMotion* motionB );
			HK_FORCE_INLINE void _resetAvgPoint();

			hkRotation m_wRmotionA;		// motionA to world rotation
			hkRotation m_wRmotionB;		// motionB to world rotation
			HK_PAD_ON_SPU( hkBool32 ) m_mergeFriction;	// If set, a single merged friction jacobian will be built (on the last manifold)


			// Note modify avgPoint and friction circle to use:
			// from: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
			// and: http://nfs-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
			// def online_variance(data):
			// n = 0
			// 	mean = 0
			// 	m_M2 = 0
			//
			// 	for x in data:
			// n = n + 1
			// 	delta = x - mean
			// 	mean = mean + delta/n
			//  delta2 = x - mean
			// 	m_M2 = m_M2 + delta*delta2
			//
			// 	variance_n = m_M2/n      # Population Variance

			HK_PAD_ON_SPU( int ) m_numPoints;
			HK_PAD_ON_SPU( hkReal ) m_previousImpulses;	///< the sum of all normal impulses from the last frame.
			hkVector4 m_contactAvgPosition;
			hkVector4 m_contactAvgNormal;
			hkVector4 m_M2;
		};


		/// Inline version of converting a manifold into Jacobians. Include hknpContactSolverSetup.inl to use this function.
		/// Notes:
		///  - solverSumVelAStream is only needed when this is called from within the solver
		template<BuildJacConfig TYPE>
		HK_FORCE_INLINE static void HK_CALL _buildContactJacobianForSingleManifold(
			const hknpSimulationThreadContext* tl,
			const hknpSolverInfo& solverInfo, const hknpManifoldCollisionCache* collisionCache, const hknpManifold* manifold, BuildConfig* HK_RESTRICT buildConfig,
			const hknpCdBodyBase& cdBodyA, const hknpSolverSumVelocity* solverSumVelAStream,
			const hknpCdBodyBase& cdBodyB, const hknpSolverSumVelocity* solverSumVelBStream,
			hknpMxContactJacobian* HK_RESTRICT mxJacobian, hknpMxContactJacobian* HK_RESTRICT mxJacobianOnPpu, int indexOfManifoldInJacobian
			);

		/// Out of line version of _buildContactJacobianForSingleManifold
		template<BuildJacConfig TYPE>
		static void HK_CALL buildContactJacobianForSingleManifold(
			const hknpSimulationThreadContext* tl,
			const hknpSolverInfo& solverInfo, const hknpManifoldCollisionCache* collisionCache, const hknpManifold* manifold, BuildConfig* HK_RESTRICT buildConfig,
			const hknpCdBody& cdBodyA, const hknpSolverSumVelocity* solverSumVelAStream,
			const hknpCdBody& cdBodyB, const hknpSolverSumVelocity* solverSumVelBStream,
			hknpMxContactJacobian* HK_RESTRICT mxJacobian, hknpMxContactJacobian* HK_RESTRICT mxJacobianOnPpu, int indexOfManifoldInJacobian
		);

		/// Simple user call to add a contact, without worrying about details like material and callbacks.
		/// You can use the hknpMxJacobianSorter to work out indexOfManifoldInJacobian
		static void HK_CALL buildContactJacobian(
			const hknpManifold& manifold,	const hknpSolverInfo& solverInfo, const BuildContactInput& input,
			const hknpBody* HK_RESTRICT bodyA, const hknpMotion* motionA, const hknpMaterial* childMaterialA,
			const hknpBody* HK_RESTRICT bodyB, const hknpMotion* motionB, const hknpMaterial* childMaterialB,
			hknpMxContactJacobian* HK_RESTRICT mxJacobianOut, hknpMxContactJacobian* HK_RESTRICT mxJacobianOnPpu, int indexOfManifoldInJacobian
			);

		///
		static void HK_CALL buildContactJacobianWithCache(
			const hknpSimulationThreadContext* tl,
			const hknpManifold& manifold,	const hknpSolverInfo& solverInfo, BuildConfig* HK_RESTRICT buildConfig,
			const hknpBody* HK_RESTRICT bodyA, const hknpMotion* motionA, const hknpMaterial* childMaterialA,
			const hknpBody* HK_RESTRICT bodyB, const hknpMotion* motionB, const hknpMaterial* childMaterialB,
			hknpMxContactJacobian* HK_RESTRICT mxJacobianOut, hknpMxContactJacobian* HK_RESTRICT mxJacobianOnPpu, int indexOfManifoldInJacobian
			);
};


#endif // HKNP_CONTACT_SOLVER_SETUP_H

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

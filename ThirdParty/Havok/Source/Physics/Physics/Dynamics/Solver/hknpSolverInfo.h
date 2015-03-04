/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SOLVER_INFO_H
#define HKNP_SOLVER_INFO_H


/// Helper struct to configure the solver.
struct hknpSolverInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSolverInfo );

		HK_ON_CPU( hknpSolverInfo() );

		void setTauAndDamping( hkReal tau, hkReal damping );

		void setStepInfo( hkVector4Parameter gravity, hkReal collisionTolerance, hkReal dt, int numSubsteps, int numMicrosteps );

		void setMaxTimDistance( hkReal d );

		HK_FORCE_INLINE hkSimdReal getRhsFactorInv() const;

	public:

		hkReal		m_tau;
		hkSimdReal	m_damping;
		hkReal		m_frictionTau;			// unused

		hkSimdReal	m_deltaTime;			///< The frame delta time
		hkSimdReal	m_invDeltaTime;			///< The frame inverted delta time

		hkSimdReal	m_subStepDeltaTime;		///< The delta time for each solver substep
		hkSimdReal	m_subStepInvDeltaTime;	///< The inverted delta time for each solver substep

		hkSimdReal	m_distanceToLinearTim;	///< Factor to convert a distance to a linear TIM
		hkSimdReal	m_linearTimToDistance;	///< Factor to convert a linear TIM to a distance

		hkSimdReal	m_collisionTolerance;

		hkSimdReal	m_unitScale;			/// See hknpWorldCinfo::m_unitScale

		hkSimdReal	m_collisionAccuracy;	/// See hknpWorldCinfo::m_relativeCollisionAccuracy

		/// The max approaching velocity for objects flagged with hknpBodyQuality::HIGH_QUALITY_SOLVER.
		
		hkReal		m_maxApproachSpeedForHighQualitySolver;

		int			m_numSteps;
		hkReal		m_invNumSteps;

		int  		m_numMicroSteps;
		hkReal		m_invNumMicroSteps;		// unused

		hkReal		m_nominalGravityLength;	// unused
		hkReal		m_expectedDeltaTime;	// unused

		hkUint32	m_stepSolveCount;		///< Counts the simulation steps. Initialized to zero.

		hkVector4	m_globalAccelerationPerSubStep;	// gravity * subDt
		hkVector4	m_globalAccelerationPerStep;	// unused

		hkVector4	m_integrateVelocityFactor;
		hkVector4	m_invIntegrateVelocityFactor;

		// Precomputed helper variables for hknpContactSolverSetup::buildContactJacobianForSingleManifold().
		hkSimdReal	m_rhsFactor;
		hkSimdReal	m_frictionRhsFactor;

		// tau/damping
		hkReal		m_dampDivTau; // unused
		hkReal		m_tauDivDamp; // unused
		hkReal		m_dampDivFrictionTau; // unused
		hkReal		m_frictionTauDivDamp; // unused
};

#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.inl>


#endif // HKNP_SOLVER_INFO_H

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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_BILATERAL_CONSTRAINT_INFO_H
#define HKP_BILATERAL_CONSTRAINT_INFO_H

class hkpSolverResults;


///
class hkp1dLinearBilateralConstraintInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp1dLinearBilateralConstraintInfo );

		HK_FORCE_INLINE hkp1dLinearBilateralConstraintInfo() {}

		/// the pivot point A in world space
		hkVector4 m_pivotA;

		/// the pivot point B in world space
		hkVector4 m_pivotB;

		/// defines the normal of a plane that movement is restricted to lay on, in world space
		hkVector4 m_constrainedDofW;
};


///
class hkp1dLinearBilateralUserTauConstraintInfo : public hkp1dLinearBilateralConstraintInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp1dLinearBilateralUserTauConstraintInfo );

			/// The stiffness of the constraint
		hkPadSpu<hkReal> m_tau;

			/// The damping of the constraint
		hkPadSpu<hkReal> m_damping;
};


///
class hkp1dAngularBilateralConstraintInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp1dAngularBilateralConstraintInfo );

		/// m_zeroErrorAxisAinW X m_zeroErrorAxisBinW = m_constrainedDofW
		HK_FORCE_INLINE hkp1dAngularBilateralConstraintInfo() {}

			/// an axis perpendicular to the constraint axis, in world space
		hkVector4 m_zeroErrorAxisAinW;

			/// defines the axis that rotational movement is not allowed along, in world space
		hkVector4 m_constrainedDofW;

			/// perpendicular to m_zeroErrorAxisAinW axis and the constraint axis, transformed from B's local space into world space
		hkVector4 m_perpZeroErrorAxisBinW;
};


///
class hkp1dLinearLimitInfo : public hkp1dLinearBilateralConstraintInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp1dLinearLimitInfo );

		hkPadSpu<hkReal> m_min;
		hkPadSpu<hkReal> m_max;
};


///
class hkp1dAngularFrictionInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp1dAngularFrictionInfo );

		hkPadSpu<const hkVector4*>      m_constrainedDofW;		// points to an array of axis
		hkPadSpu<hkpSolverResults*>      m_lastSolverResults;
		hkPadSpu<hkReal>                m_maxFrictionTorque;
		hkPadSpu<int>	                m_numFriction;				// number of frictions added
};


///
class hkp1dLinearFrictionInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp1dLinearFrictionInfo );

		/// the pivot point in world space
		hkVector4 m_pivot;

		/// defines the normal of a plane that movement is restricted to lay on, in world space
		hkVector4 m_constrainedDofW;

		hkPadSpu<hkReal> m_maxFrictionForce;

		hkPadSpu<hkpSolverResults*> m_lastSolverResults;
};


///
class hkp1dAngularLimitInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkp1dAngularLimitInfo );

			/// defines the axis that rotational movement is not allowed along, in world space
		hkVector4 m_constrainedDofW;

			/// the lower limit of angular freedom
		hkPadSpu<hkReal> m_min;

			/// the upper limit of angular freedom
		hkPadSpu<hkReal> m_max;

			/// The current angle
		hkPadSpu<hkReal> m_computedAngle;

			/// The tau used by the solver
		hkPadSpu<hkReal> m_tau;
};


/// Holds parameters needed to build Jacobians for a pulley constraint.
class hkpPulleyConstraintInfo
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpPulleyConstraintInfo );

			/// Rope attachment point (pivot point) of body A in the global space.
		hkVector4 m_positionA;

			/// Rope attachment point (pivot point) of body B in the global space.
		hkVector4 m_positionB;

			/// Pulley pivot point on the bodyA's side; in the global space.
		hkVector4 m_pulleyPivotA;

			/// Pulley pivot point on the bodyB's side; in the global space
		hkVector4 m_pulleyPivotB;

			/// Combined length of rope used ( equal to (rope on bodyA's side + rope on bodyB's side * leverageOnBodyB) )
		hkPadSpu<hkReal> m_ropeLength;

			/// Leverage on body B.
		hkPadSpu<hkReal> m_leverageOnBodyB;
};


class hkpConstraintQueryIn;
class hkpConstraintQueryOut;
class hkpSolverResults;
class hkp1Lin2AngJacobian;


extern "C"
{
	void HK_CALL hk1dLinearBilateralConstraintBuildJacobian( const hkp1dLinearBilateralConstraintInfo& info, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hk1dLinearBilateralConstraintBuildJacobianWithCustomRhs( const hkp1dLinearBilateralConstraintInfo& info, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out, hkReal customRhs );

	void HK_CALL hk1dLinearBilateralConstraintBuildJacobianWithCustomRhs_noSchema( const hkp1dLinearBilateralConstraintInfo& info, const hkpConstraintQueryIn &in, hkp1Lin2AngJacobian* HK_RESTRICT jac, hkReal customRhs );

	void HK_CALL hk1dLinearBilateralConstraintUserTauBuildJacobian( const hkp1dLinearBilateralUserTauConstraintInfo& info, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hk1dAngularBilateralConstraintBuildJacobian( const hkp1dAngularBilateralConstraintInfo& info, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hk1dLinearLimitBuildJacobian( const hkp1dLinearLimitInfo& info, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hk1dAngularLimitBuildJacobian( const hkp1dAngularLimitInfo& info, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hk1dAngularFrictionBuildJacobian( const hkp1dAngularFrictionInfo& info, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hk1dLinearFrictionBuildJacobian( const hkp1dLinearFrictionInfo& info, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	//
	// Modifiers
	//

	void HK_CALL hkSetInvMassBuildJacobian( hkVector4Parameter invMassA, hkVector4Parameter invMassB, hkpConstraintQueryOut &out );

	void HK_CALL hkAddVelocityBuildJacobian( hkVector4Parameter deltaVel, int bodyIndex, hkpConstraintQueryOut &out );

	void HK_CALL hkSetCenterOfMassBuildJacobian( const hkMatrix3& angToLinVelA, const hkMatrix3& angToLinVelB, hkpConstraintQueryOut &out );
}

HK_FORCE_INLINE void HK_CALL hkEndConstraints() {}


#endif // HKP_BILATERAL_CONSTRAINT_INFO_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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

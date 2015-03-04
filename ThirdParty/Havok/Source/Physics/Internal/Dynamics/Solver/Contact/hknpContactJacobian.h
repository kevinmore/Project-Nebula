/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_CONTACT_JACOBIAN_H
#define HKNP_CONTACT_JACOBIAN_H

#include <Physics/Physics/hknpConfig.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>

//#define HKNP_MX_FRICTION

// Forward declarations.
template <int M> class hkMxVector;

#if defined(HK_PLATFORM_WIN32)
//#	pragma warning( 3 : 4820 )		
#endif

/// Contact Jacobian types
struct hknpContactJacobianTypes
{
	struct HeaderData
	{
		hknpBodyId::Type	m_solverVelIdA;
		hknpBodyId::Type	m_solverVelIdB;
		hknpBodyId		m_bodyIdA;			// needed for solver callbacks
		hknpBodyId		m_bodyIdB;			// needed for solver callbacks
		hknpShapeKey	m_shapeKeyA;		// needed for solver callbacks
		hknpShapeKey	m_shapeKeyB;		// needed for solver callbacks

		hknpCellIndex	m_cellIndexA;		// debug only
		hknpCellIndex	m_cellIndexB;		// debug only
		hkUint8			m_manifoldType;		// the type of the manifold. See hknpManifold::ManifoldType.
		hkUint8			m_isLinkFlipped;

		hkUint8			m_numContactPoints;
		hkUint8			m_frictionEnabled;	///< if set, the friction has been initialized. This is only used if !defined(HKNP_MX_FRICTION)
		hkUint8			m_fractionOfClippedImpulseToApply; // 0(0.0f) -> No clipping; 255(1.0f) -> clip max impulse
		hkUint8			m_padding[1];

		hknpBodyFlags	m_enabledModifiers;
		hkReal			m_contactPointMinusComA_DotNormal;		  ///< = (contactPoint - comB).dot3(normal), needed to reconstruct the contact points
		hknpManifoldCollisionCache* m_collisionCacheInMainMemory; /// Points back to the collision cache. (in PPU memory in PS3).
	};

	template <int M>
	struct ContactPointData
	{
#if HKNP_CONTACT_JACOBIAN_IS_COMPRESSED
		HK_ALIGN_REAL(hkHalf m_angular[M][6]); // normal direction constraints for points, no w components.
#else
		HK_ALIGN_REAL(hkHalf m_angular[M][8]); // normal direction constraints for points
#endif
		hkReal m_effectiveMass[M]; // this is called getInvJacDiagSr too.
		hkReal m_rhs[M];
	};
};


/// Friction properties of a contact Jacobian.
struct hknpContactFrictionJacobian
{
	hkReal m_frictionEffMass[3];	// Effective mass along linear0, linear1, angular.
	hkReal m_frictionRadius[1];

	hkReal m_frictionRhs    [3];	// RHS along linear0, linear1, angular
	hkReal m_maxFrictionImpulse[1];

	hkHalf m_deltaFictionRhs[3];	// RHS along linear0, linear1, angular
	hkHalf m_padding;

	// Each four of hkHalfs from hkHalf arrays stores
	HK_ALIGN_REAL(hkHalf) m_friction01Lin0[8]; // Linear friction; linear jacobians for body A for both friction '0' and friction '1'.
	hkHalf m_friction0Ang01[8];			// Linear friction '0'; angular Jacobians for both bodies.
	hkHalf m_friction1Ang01[8];
	hkHalf m_frictionAngularAng01[8];	// central torque friction, angular Jacobians for both bodies.
};


/// This Jacobian holds contact points from several independent manifolds.
/// Each manifold 'links' different bodies, not referenced by any of the other manifolds from this Jacobian.
/// Each manifold can generate up to four points plus 3D friction.
template <int M>
struct hknpContactJacobian
{
	public:

		typedef hknpContactJacobianTypes::HeaderData			ManifoldData;
		typedef hknpContactJacobianTypes::ContactPointData<M>	ContactPointData;

		enum
		{
			NUM_MANIFOLDS = M,
			NUM_POINTS_PER_MANIFOLD = 4,
			IS_COMPRESSED = HKNP_CONTACT_JACOBIAN_IS_COMPRESSED,	// Only supported with NUM_MANIFOLDS = 4.
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpContactJacobian<M> );

		/// Zero all Jacobians for chosen manifold only.
		template <int manifoldIndex> HK_FORCE_INLINE void zeroManifold();
		HK_FORCE_INLINE void zeroManifold(int manifoldIndex);

		HK_FORCE_INLINE void prefetch();

		HK_FORCE_INLINE int getSizeInBytes() const { return hkSizeOf(hknpContactJacobian<M>); }

		template <int P>
		HK_FORCE_INLINE static int HK_CALL calcSizeInBytes() { return hkSizeOf(hknpContactJacobian<M>) /*+ ( P * hkSizeOf(hknpContactJacobian<M>::PerContactPointData) )*/; } // todo var num points

		/// Pack all contact points of a manifold at once.
		template <hkMathRoundingMode A, int NUM_MX>
		HK_FORCE_INLINE void packManifoldAngular( int manifoldIndex, int contactPointIdx, const hkMxVector<NUM_MX>& angular0, const hkMxVector<NUM_MX>& angular1 );

		/// Unpack all contact points of a manifold at once.
		HK_FORCE_INLINE void unpackManifoldAngular( int manifoldIndex, hkMxVector<NUM_POINTS_PER_MANIFOLD>* HK_RESTRICT angular0, hkMxVector<NUM_POINTS_PER_MANIFOLD>* HK_RESTRICT angular1 ) const;

		/// Unpack all manifolds of a contact point at once for any NUM_MANIFOLDS.
		HK_FORCE_INLINE void unpackContactPointAngular( int pointIndex, hkMxVector<NUM_MANIFOLDS>* HK_RESTRICT angular0, hkMxVector<NUM_MANIFOLDS>* HK_RESTRICT angular1 ) const;

		/// Unpack all manifolds of a contact point at once for NUM_MANIFOLDS=1.
		HK_FORCE_INLINE void unpackContactPointAngular( int pointIndex, hkVector4* HK_RESTRICT angular0, hkVector4* HK_RESTRICT angular1 ) const;

		/// Copy the given manifold from the given Jacobian.
		template <int N>
		HK_FORCE_INLINE void copyFrom( const hknpContactJacobian<N>& srcJac, int srcManifoldIdx, int dstManifoldIdx );

	public:

#if defined(HK_PLATFORM_HAS_SPU)
		HK_ALIGN_REAL(hknpContactJacobianTypes::HeaderData m_manifoldData[M]);	// align to help SPU code layout
#else
		hknpContactJacobianTypes::HeaderData m_manifoldData[M];
#endif
		hkUint8 m_disableNegImpulseClip;	// if bit 1<<mxIndex set to one, negative impulse clipping is disabled
		hkUint8 m_useIncreasedIterations;	// if bit 1<<mxIndex set to one, extra iterations will be performed
		hkUint8 m_maxNumContactPoints;		// maximum number of contact points in the current manifolds
		hkUint8 m_executeLastIterationOnly;	///< Set this if you are only interested in the last iteration.

		// Each of these member variables is read as a coalesced batch, so they are not inside the header
		// Note: they have to be added to jacobian copy function manually as they are not part of the header data
		hkReal m_maxImpulsePerStep[M];
		HK_ALIGN_REAL(hkReal m_massChangerData[M]);	// object A invMass will be multiplied with 1+m_massChangerData, object b invMass with 1+m_massChangerData
		hkVector4 m_linear0[M];			// Each linear Jacobian is shared for all contact points in a given manifold
		HK_ALIGN_REAL( ContactPointData m_contactPointData[NUM_POINTS_PER_MANIFOLD] );

		//
		// Friction related stuff
		//
#if defined(HKNP_MX_FRICTION)
		hkReal m_frictionRadius[M];
		hkReal m_frictionCoefficient[M];

		hkReal m_frictionEffMass[3][M]; // Effective mass along linear0, linear1, angular.
		hkReal m_frictionRhs    [3][M];	// RHS along linear0, linear1, angular
		hkHalf m_deltaFictionRhs[3][M];	// RHS along linear0, linear1, angular

		// Each four of hkHalfs from hkHalf arrays stores
		HK_ALIGN_REAL(hkHalf) m_friction01Lin0[8*M]; // Linear friction; linear jacobians for body A for both friction '0' and friction '1'.
		hkHalf m_friction0Ang01[8*M]; // Linear friction '0'; angular Jacobians for both bodies.
		hkHalf m_friction1Ang01[8*M];
		hkHalf m_frictionAngularAng01[8*M];	// central torque friction, angular Jacobians for both bodies.
#else
		hknpContactFrictionJacobian m_friction[M];
#endif
};


#if defined(HK_PLATFORM_WIN32)
//#	pragma warning( disable : 4820 )
#endif


// Helper class which points to an offset version of hknpMxContactJacobian to avoid repeated index calculations
#if defined(HKNP_MX_FRICTION)
template <int MXLENGTH>
struct hknpMxContactJacobianFloats : public hknpContactJacobian<MXLENGTH>
	{
	};
#endif

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.inl>


#endif // HKNP_CONTACT_JACOBIAN_H

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

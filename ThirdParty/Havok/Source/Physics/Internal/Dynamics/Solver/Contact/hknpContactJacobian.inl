/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Math/Vector/hkHalf8Util.h>
#include <Common/Base/Math/Vector/Mx/hkMxHalf8.h>
#include <Common/Base/Math/Vector/Mx/hkMxVector.h>
#include <Physics/Physics/hknpTypes.h>


#if (HKNP_CONTACT_JACOBIAN_IS_COMPRESSED == 1)
	HK_COMPILE_TIME_ASSERT( hknpMxContactJacobian::NUM_MANIFOLDS == 4 );	// Supporting multiples of 4 is quite easy, supporting random counts is more complicated
	HK_COMPILE_TIME_ASSERT( hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD == 4 );
#endif


template <int MXLENGTH>
template <int manifoldIndex>
HK_FORCE_INLINE void hknpContactJacobian<MXLENGTH>::zeroManifold()
{
	HK_COMPILE_TIME_ASSERT( manifoldIndex < MXLENGTH );
	zeroManifold(manifoldIndex);
}

template <int MXLENGTH>
HK_FORCE_INLINE void hknpContactJacobian<MXLENGTH>::zeroManifold(int manifoldIndex)
{
	HK_ASSERT2(0x4fff7858, manifoldIndex < NUM_MANIFOLDS, "illegal manifold index" );

	// We will simply zero the virtualMass
	for (int i=0; i<NUM_POINTS_PER_MANIFOLD; ++i)
	{
		m_contactPointData[i].m_effectiveMass[manifoldIndex] = 0.0f;
	}

#if defined(HKNP_MX_FRICTION)
	m_frictionEffMass[0][manifoldIndex] = 0.0f;
	m_frictionEffMass[1][manifoldIndex] = 0.0f;
	m_frictionEffMass[2][manifoldIndex] = 0.0f;
#else
	hknpContactFrictionJacobian* jr = &m_friction[manifoldIndex];
	jr->m_frictionEffMass[0] = 0.0f;
	jr->m_frictionEffMass[1] = 0.0f;
	jr->m_frictionEffMass[2] = 0.0f;
#endif
	m_manifoldData[manifoldIndex].m_manifoldSolverInfo = HK_NULL;
}

template <int MXLENGTH>
HK_FORCE_INLINE void hknpContactJacobian<MXLENGTH>::prefetch()
{
	hkMath::forcePrefetch<sizeof(hknpContactJacobian<MXLENGTH>)>(this);
	hkMath::forcePrefetch<sizeof(hknpContactJacobian<MXLENGTH>)>(hkAddByteOffset( this, 256 ));
	hkMath::forcePrefetch<sizeof(hknpContactJacobian<MXLENGTH>)>(hkAddByteOffset( this, 512 ));
	hkMath::forcePrefetch<sizeof(hknpContactJacobian<MXLENGTH>)>(hkAddByteOffset( this, 768 ));
}


// Helper class for compressed contact jacobians.
struct hknpContactJacobianCompressedUtil
{
	// Packs the (x,y,z) components of each vector in the pair (v0,v1) into a packed half array at the specified pairIndex in the range [0,4].
	// The first pair of vectors (v0, v1) has pairIndex 0, the second pair has pairIndex 1, and so on.
	template<hkMathRoundingMode A>
	static HK_FORCE_INLINE void pack( hkVector4Parameter v0, hkVector4Parameter v1, hkVector4* HK_RESTRICT packedHalfArray, int pairIndex )
	{
		hkVector4 packed;
		hkHalf8Util::packInterleaved<A>( v0, v1, packed );

		hkSimdReal yy = packed.getComponent<1>();
		hkSimdReal zz = packed.getComponent<2>();

		packed.store<1>( &packedHalfArray[0](pairIndex) );
		yy.store<1>( &packedHalfArray[1](pairIndex) );
		zz.store<1>( &packedHalfArray[2](pairIndex) );
	}

	// Unpacks the (x,y,z) components of all four vector pairs from a packed half array.
	static HK_FORCE_INLINE void unpack( const hkVector4* HK_RESTRICT packedHalfArray, hkVector4* HK_RESTRICT v0, hkVector4* HK_RESTRICT v1 )
	{
		hkVector4 p0 = packedHalfArray[0];
		hkVector4 p1 = packedHalfArray[1];
		hkVector4 p2 = packedHalfArray[2];
		hkVector4 p3 = packedHalfArray[2];
		HK_TRANSPOSE4( p0, p1, p2, p3 );

		hkHalf8Util::unpackInterleaved( p0, v0+0, v1+0);
		hkHalf8Util::unpackInterleaved( p1, v0+1, v1+1);
		hkHalf8Util::unpackInterleaved( p2, v0+2, v1+2);
		hkHalf8Util::unpackInterleaved( p3, v0+3, v1+3);
	}

	// Unpacks the (x,y,z) components of one vector pair at the specified pairIndex from a packed half array.
	static HK_FORCE_INLINE void unpack( const hkVector4* HK_RESTRICT packedHalfArray, hkVector4* HK_RESTRICT v0, hkVector4* HK_RESTRICT v1, int pairIndex )
	{
		hkSimdReal x; x.load<1>( &packedHalfArray[0](pairIndex) );
		hkSimdReal y; y.load<1>( &packedHalfArray[1](pairIndex) );
		hkSimdReal z; z.load<1>( &packedHalfArray[2](pairIndex) );
		hkVector4 p; p.set( x,y,z,z );
		hkHalf8Util::unpackInterleaved( p, v0, v1);
	}
};


template <int MXLENGTH>
template <hkMathRoundingMode A, int NUM_MX>
#if defined( HK_ARCH_IA32 ) && defined( HK_PLATFORM_WIN32 ) && defined( HK_COMPILER_MSVC ) && ( _MSC_VER == 1600 ) && defined( HK_DEBUG )
/* Inlining on Win32 VS2010 debug config causes ICE */
#else
HK_FORCE_INLINE
#endif
void hknpContactJacobian<MXLENGTH>::packManifoldAngular(
	int manifoldIndex, int contactPointIdx,
	const hkMxVector<NUM_MX>& angular0, const hkMxVector<NUM_MX>& angular1 )
{
	if ( IS_COMPRESSED )
	{
		HK_ASSERT( 0xf04ff790, NUM_MX==4 && contactPointIdx == 0);
		hkVAR_UNROLL(NUM_MX,
			hknpContactJacobianCompressedUtil::pack<A>(
				angular0.template getVector<hkI>(),
				angular1.template getVector<hkI>(),
				(hkVector4* HK_RESTRICT) &m_contactPointData[0].m_angular,
				manifoldIndex
			)
		);
	}
	else
	{
		hkMxHalf8<NUM_MX> packedAngularJac; packedAngularJac.pack( angular0, angular1 );
		packedAngularJac.template scatter<sizeof(hknpMxContactJacobian::ContactPointData)>( &m_contactPointData[contactPointIdx].m_angular[manifoldIndex][0] );
	}
}

template <int MXLENGTH>
HK_FORCE_INLINE void hknpContactJacobian<MXLENGTH>::unpackManifoldAngular( int manifoldIndex, hkMxVector<NUM_POINTS_PER_MANIFOLD>* HK_RESTRICT angular0, hkMxVector<NUM_POINTS_PER_MANIFOLD>* HK_RESTRICT angular1 ) const
{
	if ( IS_COMPRESSED )
	{
		hkMxUNROLL_4(
			hknpContactJacobianCompressedUtil::unpack(
				(hkVector4* HK_RESTRICT) &m_contactPointData[hkMxI].m_angular,
				&angular0->m_vec.v[hkMxI],
				&angular1->m_vec.v[hkMxI],
				manifoldIndex
			)
		);
	}
	else
	{
		hkMxUNROLL_4(
			hkHalf8Util::unpack<HK_IO_SIMD_ALIGNED>(
				(const hkHalf* HK_RESTRICT)&m_contactPointData[hkMxI].m_angular[manifoldIndex],
				&angular0->m_vec.v[hkMxI],
				&angular1->m_vec.v[hkMxI]
			)
		);
	}
}

template <int MXLENGTH>
HK_FORCE_INLINE void hknpContactJacobian<MXLENGTH>::unpackContactPointAngular( int pointIndex, hkMxVector<NUM_MANIFOLDS>* HK_RESTRICT angular0, hkMxVector<NUM_MANIFOLDS>* HK_RESTRICT angular1 ) const
{
	if ( IS_COMPRESSED )
	{
		hknpContactJacobianCompressedUtil::unpack( (const hkVector4* HK_RESTRICT) m_contactPointData[pointIndex].m_angular, angular0->m_vec.v, angular1->m_vec.v );
	}
	else
	{
		reinterpret_cast<const hkMxHalf8<NUM_MANIFOLDS>*>(m_contactPointData[pointIndex].m_angular)->unpack(*angular0, *angular1);
	}
}


template <int MXLENGTH>
HK_FORCE_INLINE void hknpContactJacobian<MXLENGTH>::unpackContactPointAngular( int pointIndex, hkVector4* HK_RESTRICT angular0, hkVector4* HK_RESTRICT angular1 ) const
{
	HK_COMPILE_TIME_ASSERT( IS_COMPRESSED == 0 );

	angular0->load<4,HK_IO_NATIVE_ALIGNED>(&m_contactPointData[pointIndex].m_angular[0][0]);
	angular1->load<4,HK_IO_NATIVE_ALIGNED>(&m_contactPointData[pointIndex].m_angular[0][4]);
}

//
//	Copies the given manifold from the given Jacobian

template <int M>
template <int N>
HK_FORCE_INLINE void hknpContactJacobian<M>::copyFrom(const hknpContactJacobian<N>& srcJac, int srcManifoldIdx, int dstManifoldIdx)
{
	// Copy manifold data
	m_manifoldData[dstManifoldIdx] = srcJac.m_manifoldData[srcManifoldIdx];

	// Copy flags from source manifold to destination manifold
	m_disableNegImpulseClip		&= ~(1 << dstManifoldIdx);
	m_useIncreasedIterations	&= ~(1 << dstManifoldIdx);
	m_disableNegImpulseClip		|= ((srcJac.m_disableNegImpulseClip >> srcManifoldIdx) & 1) << dstManifoldIdx;
	m_useIncreasedIterations	|= ((srcJac.m_useIncreasedIterations >> srcManifoldIdx) & 1) << dstManifoldIdx;

	// Per-manifold stuff
	m_maxImpulsePerStep		[dstManifoldIdx]	= srcJac.m_maxImpulsePerStep	[srcManifoldIdx];
	m_massChangerData		[dstManifoldIdx]	= srcJac.m_massChangerData		[srcManifoldIdx];
	m_linear0				[dstManifoldIdx]	= srcJac.m_linear0				[srcManifoldIdx];

#if defined(HKNP_MX_FRICTION)
	m_frictionRadius		[dstManifoldIdx]	= srcJac.m_frictionRadius		[srcManifoldIdx];
	m_frictionCoefficient	[dstManifoldIdx]	= srcJac.m_frictionCoefficient	[srcManifoldIdx];

	for (int k = 0; k < 3; k++)
	{
		m_frictionEffMass	[k][dstManifoldIdx]	= srcJac.m_frictionEffMass	[k][srcManifoldIdx];
		m_frictionRhs		[k][dstManifoldIdx]	= srcJac.m_frictionRhs		[k][srcManifoldIdx];
		m_deltaFictionRhs	[k][dstManifoldIdx]	= srcJac.m_deltaFictionRhs	[k][srcManifoldIdx];
	}

	// Friction
	hkString::memCpy(&m_friction01Lin0			[dstManifoldIdx << 3], &srcJac.m_friction01Lin0			[srcManifoldIdx << 3], sizeof(hkHalf) << 3);
	hkString::memCpy(&m_friction0Ang01			[dstManifoldIdx << 3], &srcJac.m_friction0Ang01			[srcManifoldIdx << 3], sizeof(hkHalf) << 3);
	hkString::memCpy(&m_friction1Ang01			[dstManifoldIdx << 3], &srcJac.m_friction1Ang01			[srcManifoldIdx << 3], sizeof(hkHalf) << 3);
	hkString::memCpy(&m_frictionAngularAng01	[dstManifoldIdx << 3], &srcJac.m_frictionAngularAng01	[srcManifoldIdx << 3], sizeof(hkHalf) << 3);
#else
	hkString::memCpy16(&m_friction[dstManifoldIdx], &srcJac.m_friction[srcManifoldIdx], sizeof( hknpContactFrictionJacobian)>>4 );
#endif



	m_maxNumContactPoints = hkMath::max2(m_maxNumContactPoints, srcJac.m_manifoldData[srcManifoldIdx].m_numContactPoints);

	// Contacts
	typedef hknpContactJacobianTypes::ContactPointData<N>	SrcContactPointData;
	typedef hknpContactJacobianTypes::ContactPointData<M>	DstContactPointData;
	for (int k = 0; k < 4; k++)
	{
		const SrcContactPointData& srcCp	= srcJac.m_contactPointData[k];
		DstContactPointData& dstCp			= m_contactPointData[k];

		dstCp.m_effectiveMass	[dstManifoldIdx]	= srcCp.m_effectiveMass	[srcManifoldIdx];
		dstCp.m_rhs		[dstManifoldIdx]	= srcCp.m_rhs		[srcManifoldIdx];

		const hkHalf* HK_RESTRICT srcAng	= srcCp.m_angular[srcManifoldIdx];
		hkHalf* HK_RESTRICT dstAng			= dstCp.m_angular[dstManifoldIdx];

#if HKNP_CONTACT_JACOBIAN_IS_COMPRESSED
		hkString::memCpy(dstAng, srcAng, sizeof(hkHalf) * 6);
#else
		hkString::memCpy(dstAng, srcAng, sizeof(hkHalf) << 3);
 #endif
	}
}

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

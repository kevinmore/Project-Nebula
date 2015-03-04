/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_CONTINUOUS_GSK
#define HK_COLLIDE2_CONTINUOUS_GSK

#include <Physics2012/Collide/Agent3/hkpAgent3.h>
#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>

	// this is minimal timestep. This only affects collisions if
	// a point of an object 
	// at time x				is outside the m_toiDistance
	// at time x+minTimeStep    is outside the m_toiDistance
	// but in between           its inside m_minSeparation
	//
	// This can only happen if you have very high angular velocities
	// Note: Ipion used a value of 1e-3f
class hkpGskCache;

struct hkp4dGskVertexCollidePointsInput
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CDINFO, hkp4dGskVertexCollidePointsInput );

	hkPadSpu<const hkMotionState*>	m_motionA;
	hkPadSpu<const hkMotionState*>	m_motionB;

	hkPadSpu<hkVector4*>			m_verticesA;
	hkPadSpu<int>					m_numVertices;
	hkPadSpu<int>					m_allocatedNumVertices;

	hkPadSpu<hkReal>				m_radiusSum;
	hkPadSpu<hkReal>				m_maxAccel;
	hkPadSpu<hkReal>				m_invMaxAccel;

	hkVector4						m_planeB;		 // in B space
	hkVector4						m_pointOnPlaneB; // in B space

	hkPadSpu<const hkStepInfo*>		m_stepInfo;

	hkPadSpu<hkReal>				m_worstCaseApproachingDelta;

	hkVector4						m_linearTimInfo;
	hkVector4						m_deltaAngles[2];

	hkPadSpu<hkReal>				m_startRt;	// the relative time to start 
};

struct hkp4dGskVertexCollidePointsOutput
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CDINFO, hkp4dGskVertexCollidePointsOutput );

	hkPadSpu<hkReal> m_Rtoi;
	hkVector4 m_collidingPoint;

	hkp4dGskVertexCollidePointsOutput(): m_Rtoi(1.0f){} //m_toi( hkTime(HK_REAL_MAX)){}
};

struct hkpProcessCollisionOutput;

struct hkp4dGskTolerances
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CDINFO, hkp4dGskTolerances );

	hkPadSpu<hkReal> m_toiSeparation;
	hkPadSpu<hkReal> m_minSeparation;

	hkPadSpu<hkReal> m_minSafeDeltaRt;	// the minimum timestep scaled to [0..1]
	hkPadSpu<hkReal> m_minToiDeltaTime;	    // = 1/maxToiFrequency

	hkPadSpu<hkReal> m_toiAccuracy;
};


extern "C"
{
	void hk4dGskCollidePointsWithPlane( const hkp4dGskVertexCollidePointsInput& input, const hkp4dGskTolerances& tol, hkp4dGskVertexCollidePointsOutput& out );

		/// Find the time of impact.
		/// Rules:
		///     - If we detect a TOI, we return the time when the distance is exactly toiSeparation
		///     - We will not miss a TOI if the distance goes below minSeparation
		///     - That means minSeparation < toiSeparation < 0.0f
	void hk4dGskCollideCalcToi( const hkpAgent3ProcessInput& in3, hkSimdRealParameter allowedPenetrationDepth, hkSimdRealParameter minSeparation, hkSimdRealParameter toiSeparation, hkpGskCache& gskCache, hkVector4& separatingNormal, hkpProcessCollisionOutput& output );
}



#endif // HK_COLLIDE2_CONTINUOUS_GSK

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

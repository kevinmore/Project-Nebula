/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyQualityLibrary.h>


hknpBodyQualityLibrary::hknpBodyQualityLibrary()
{

}

hknpBodyQualityLibrary::hknpBodyQualityLibrary( hkFinishLoadedObjectFlag flag ) : hkReferencedObject(flag)
{

}

void hknpBodyQualityLibrary::initialize( const hknpBodyQualityLibraryCinfo& cinfo /*= hknpBodyQualityLibraryCinfo() */ )
{
	// Initialize all qualities and set up priorities
	for( int i = 0; i < hknpBodyQualityId::MAX_NUM_QUALITIES; i++ )
	{
		hknpBodyQuality& q = m_qualities[i];
		q.initialize( cinfo.m_unitScale );
		q.m_priority = i * 10;
	}

	//
	// Override presets
	//

	typedef hknpBodyQuality BQ;

	// Debris
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::DEBRIS ];
		q.m_supportedFlags.clear( BQ::CLIP_ANGULAR_VELOCITY | BQ::USE_HIGHER_QUALITY_CONTACT_SOLVING | BQ::ENABLE_SILHOUETTE_MANIFOLDS );
		q.m_requestedFlags.orWith( BQ::USE_DISCRETE_AABB_EXPANSION | BQ::ALLOW_CONCAVE_TRIANGLE_COLLISIONS );
		q.m_requestedFlags.orWith( BQ::MERGE_FRICTION_JACOBIANS );
	}

	// Default dynamic
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::DYNAMIC ];
		q.m_requestedFlags.orWith( BQ::USE_DISCRETE_AABB_EXPANSION  );
		if( cinfo.m_useWeldingForDefaultObjects )
		{
#if !defined(HK_PLATFORM_HAS_SPU)
			q.m_requestedFlags.orWith( BQ::ENABLE_NEIGHBOR_WELDING );
#else
			q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
#endif
		}
		else
		{
			q.m_requestedFlags.orWith( BQ::USE_DISCRETE_AABB_EXPANSION | BQ::ALLOW_CONCAVE_TRIANGLE_COLLISIONS );
		}
	}

	// WELDED SIMPLE
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::NEIGHBOR_WELDING ];
		q.m_requestedFlags.orWith( BQ::USE_DISCRETE_AABB_EXPANSION );
#if !defined(HK_PLATFORM_HAS_SPU)
		q.m_requestedFlags.orWith( BQ::ENABLE_NEIGHBOR_WELDING );
#else
		q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
#endif
	}

	// WELDED TRIANGLE
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::TRIANGLE_WELDING ];
		q.m_requestedFlags.orWith( BQ::USE_DISCRETE_AABB_EXPANSION );
#if !defined(HK_PLATFORM_HAS_SPU)
		q.m_requestedFlags.orWith( BQ::ENABLE_TRIANGLE_WELDING );
#else
		q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
#endif
	}

	// WELDED MOTION
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::MOTION_WELDING ];
		q.m_requestedFlags.orWith( BQ::USE_DISCRETE_AABB_EXPANSION );
		q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
	}

	// Critical
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::CRITICAL ];
		q.m_requestedFlags.orWith( BQ::USE_HIGHER_QUALITY_CONTACT_SOLVING );
		q.m_requestedFlags.orWith( BQ::CLIP_ANGULAR_VELOCITY );

		if( cinfo.m_useWeldingForDefaultObjects )
		{
#if !defined(HK_PLATFORM_HAS_SPU)
			q.m_requestedFlags.orWith( BQ::ENABLE_NEIGHBOR_WELDING );
#else
			q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
#endif
		}
	}

	// Vehicle
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::VEHICLE ];
		q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
	}

	// Character
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::CHARACTER ];
#if !defined(HK_PLATFORM_HAS_SPU)
		q.m_requestedFlags.orWith( BQ::ENABLE_NEIGHBOR_WELDING );
#else
		q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
#endif
	}

	// Grenade
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::GRENADE ];

#if !defined(HK_PLATFORM_HAS_SPU)
		q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING | BQ::ENABLE_NEIGHBOR_WELDING );
#else
		q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
#endif
		q.m_requestedFlags.orWith( BQ::CLIP_ANGULAR_VELOCITY ); // /*|BQ::SILHOUETTE_COLLISIONS*/
	}

	// Projectile
	{
		hknpBodyQuality& q = m_qualities[ hknpBodyQualityId::PROJECTILE ];

		q.m_requestedFlags.orWith( BQ::ENABLE_MOTION_WELDING );
		q.m_requestedFlags.orWith( BQ::CLIP_ANGULAR_VELOCITY );
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

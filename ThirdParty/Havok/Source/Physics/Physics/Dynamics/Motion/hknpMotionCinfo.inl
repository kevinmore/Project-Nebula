/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE void hknpMotionCinfo::initializeFromBody( const hknpBody& body, hkReal massOrNegativeDensity )
{
	hknpBodyCinfo cinfo;
	cinfo.m_shape = body.m_shape;
	cinfo.m_position = body.getTransform().getTranslation();
	cinfo.m_orientation.set( body.getTransform().getRotation() );
	if( massOrNegativeDensity > 0.0f )
	{
		initializeWithMass( &cinfo, 1, massOrNegativeDensity );
	}
	else if( massOrNegativeDensity < 0.0f )
	{
		initializeWithDensity( &cinfo, 1, -massOrNegativeDensity );
	}
	else
	{
		initializeAsKeyFramed( &cinfo, 1);
	}
}

HK_FORCE_INLINE void hknpMotionCinfo::initializeFromBody( const hknpBody& body )
{
	hknpBodyCinfo cinfo;
	cinfo.m_shape = body.m_shape;
	cinfo.m_position = body.getTransform().getTranslation();
	cinfo.m_orientation.set( body.getTransform().getRotation() );
	initialize( &cinfo, 1 );
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

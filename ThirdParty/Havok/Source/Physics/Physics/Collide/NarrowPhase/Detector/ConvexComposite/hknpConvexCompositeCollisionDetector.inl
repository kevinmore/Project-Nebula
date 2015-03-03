/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Types/Geometry/IntSpaceUtil/hkIntSpaceUtil.h>

HK_FORCE_INLINE void hknpConvexCompositeCollisionDetector::buildExpandedLocalSpaceAabb(
	const hknpInternalCollideSharedData &sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	hkAabb* HK_RESTRICT localSpaceAabbBodyAOut )
{
	hkAabb worldSpaceAabbBodyA; sharedData.m_intSpaceUtil->restoreAabb( cdBodyA.m_body->m_aabb, worldSpaceAabbBodyA );
	hkAabbUtil::transformAabbIntoLocalSpace( cdBodyB.m_body->getTransform(), worldSpaceAabbBodyA, *localSpaceAabbBodyAOut );
	if( cdBodyB.m_body->isDynamic() )
	{
		hkVector4 com = cdBodyA.m_motion->getCenterOfMassInWorld();
		hkVector4 linVel; cdBodyB.m_motion->_getPointVelocity( com, linVel ); linVel.setNeg<4>( linVel );
		linVel._setRotatedInverseDir( cdBodyB.m_body->getTransform().getRotation(), linVel );
		hkAabbUtil::expandAabbByMotion( *localSpaceAabbBodyAOut, linVel, *localSpaceAabbBodyAOut );
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

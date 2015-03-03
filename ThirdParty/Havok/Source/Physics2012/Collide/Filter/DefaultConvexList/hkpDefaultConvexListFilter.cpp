/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Filter/DefaultConvexList/hkpDefaultConvexListFilter.h>

hkpConvexListFilter::ConvexListCollisionType hkpDefaultConvexListFilter::getConvexListCollisionType(	const hkpCdBody& convexListBody, 
																									const hkpCdBody& otherBody, 
																									const hkpCollisionInput& input ) const
{
	
	hkpShapeType shapeType = otherBody.getShape()->getType();

	// If the convex list shape is colliding with a landscape, we dispatch the convex list shape as a list, to ensure
	// correct welding happens. Otherwise we dispatch convex list shape as normal.
	if ( input.m_dispatcher->hasAlternateType( shapeType, hkcdShapeType::BV_TREE ) )
	{
		return TREAT_CONVEX_LIST_AS_LIST;
	}
	else
	{
		return TREAT_CONVEX_LIST_AS_NORMAL;
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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/hkpShapeBase.h>
#include <Physics2012/Collide/Util/ShapeVirtualTable/hkpShapeVirtualTableUtil.h>


#if	!defined(HK_PLATFORM_SPU) && defined(HK_DEBUG)
#	define ON_UNREGISTERED_FUNCTION(fun)	hkcdShapeVirtualTableUtil::s_unregisteredFunctions |= fun
#else
#	define ON_UNREGISTERED_FUNCTION(fun)	HK_ASSERT3( 0xf032fe45, false, "The shape " << getType() << " has not registered function " << fun )
#endif


#ifndef HK_PLATFORM_SPU

hkpShapeBase::hkpShapeBase(class hkFinishLoadedObjectFlag flag)
:	hkcdShape(flag)
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpShapeBase));
	}
}

#endif

void hkpShapeBase::getAabb( const hkTransform& localToWorld, hkReal tolerance, hkAabb& aabbOut ) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_getAabb );
}

hkBool hkpShapeBase::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& output) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_castRay );
	return false;
}

void hkpShapeBase::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_castRayWithCollector );
}

hkVector4Comparison hkpShapeBase::castRayBundle(const hkpShapeRayBundleCastInput& input, hkpShapeRayBundleCastOutput& output,  hkVector4ComparisonParameter mask) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_castRayBundle );
	hkVector4Comparison cmp;
	cmp.set<hkVector4ComparisonMask::MASK_NONE>();
	return cmp;
}

void hkpShapeBase::getSupportingVertex( hkVector4Parameter direction, hkcdVertex& supportingVertexOut ) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_getSupportingVertex );
}

void hkpShapeBase::convertVertexIdsToVertices( const hkpVertexId* ids, int numIds, hkcdVertex* vertexArrayOut ) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_convertVertexIdsToVertices );
}

void hkpShapeBase::getCentre( hkVector4& centreOut ) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_getCentre );
}

int hkpShapeBase::getNumCollisionSpheres() const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_getNumCollisionSpheres );
	return 0;
}

const hkSphere* hkpShapeBase::getCollisionSpheres( hkSphere* sphereBuffer ) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_getCollisionSpheres );
	return HK_NULL;
}

#ifdef HK_PLATFORM_SPU

const hkpShape* hkpShapeBase::getChildShape( hkpShapeKey key, hkpShapeBuffer& buffer ) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_getChildShape );
	return HK_NULL;
}

hkUint32 hkpShapeBase::getCollisionFilterInfo( hkpShapeKey key ) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_getCollisionFilterInfo );
	return 0;
}

#endif

int hkpShapeBase::weldContactPoint( hkpVertexId* featurePoints, hkUint8& numFeaturePoints, hkVector4& contactPointWs, const hkTransform* thisObjTransform, const class hkpConvexShape* collidingConvexShape, const hkTransform* collidingTransform, hkVector4& separatingNormalInOut ) const
{
	ON_UNREGISTERED_FUNCTION( hkpShapeVirtualTableUtil::FUNCTION_ID_weldContactPoint );
	return 0;
}

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

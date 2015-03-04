/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>
#include <Common/Base/Types/Geometry/Sphere/hkSphere.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayBundleCastInput.h>

#if HK_POINTER_SIZE==4
HK_COMPILE_TIME_ASSERT( sizeof(hkpShape) == 16 );
#endif


#if !defined(HK_PLATFORM_SPU)

hkpShape::hkpShape( class hkFinishLoadedObjectFlag flag )
:	hkpShapeBase(flag)
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpShape));
	}
}

hkReal hkpShape::getMaximumProjection( const hkVector4& direction ) const
{
	hkTransform localToWorld;
	localToWorld.setIdentity();
	const hkReal tolerance = 0.f;
	hkAabb aabb;
	getAabb( localToWorld, tolerance, aabb);
	
	hkVector4 halfExtents; aabb.getHalfExtents( halfExtents );
	hkVector4 center; 	   aabb.getCenter( center );
	
	halfExtents.setFlipSign(halfExtents, direction);
	halfExtents.add(center);

	const hkReal result = halfExtents.dot<3>( direction ).getReal();
	return result;
}

int hkpShape::calcSizeForSpu( const CalcSizeForSpuInput& input, int spuBufferSizeLeft ) const
{
	return -1;
}

#endif


hkVector4Comparison hkpShape::castRayBundle(const hkpShapeRayBundleCastInput& input, hkpShapeRayBundleCastOutput& output, hkVector4ComparisonParameter mask) const
{
	hkVector4 start[4];
	hkVector4 end[4];
	input.m_from.extract(start[0], start[1], start[2], start[3]);
	input.m_to.extract(end[0], end[1], end[2], end[3]);

	hkpShapeRayCastInput shapeInput;
	shapeInput.m_filterInfo = input.m_filterInfo;
	shapeInput.m_rayShapeCollectionFilter = input.m_rayShapeCollectionFilter;

	int componentFlags = 0;

	for (int i=0 ; i < 4; i++)
	{
		if ( mask.anyIsSet(hkVector4Comparison::getMaskForComponent(i)) )
		{
			shapeInput.m_from = start[i];
			shapeInput.m_to = end[i];
			hkBool hit = castRay( shapeInput, output.m_outputs[i] );

			if (hit)
			{
				componentFlags |= (int) hkVector4Comparison::getMaskForComponent(i);
			}
		}
	}

	hkVector4Comparison hitMask; hitMask.set((hkVector4ComparisonMask::Mask)componentFlags);
	return hitMask;
}


/*! \fn hkBool hkpShape::castRay( const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
* Generally we recommend that you use the hkpWorld or hkpPhantom castRay() functions for raycasting.
* Always finds the closest hit and only reports this single hit.
* Returns 0 if there is no hit and 1 if there is a hit.
* The following rules apply:
* 1) A startpoint does not return a hit if the startpoint is in contact with the surface of an object and 
* the ray does not intersect the object. 
* - The exception to this rule is hkpTriangleShape which DOES return a hit in rule 1) above.
* 2) If a ray is parallel and exactly tangental to the objects geometric surface, it won't hit. One exception to this rule is hkpCapsuleShape which does return a hit.
* 3) If the start point of a ray is inside the object, it won't hit
* 4) It only returns a hit, if the new m_hitFraction is less than the current results.m_hitFraction 
*   which should be initialized with 1.0f)
* 5) It returns true if it hits
* 6) If it hits, than it sets the result.m_hitFraction to less than 1.0f, it does not set it if it does not hit.	
*/

/*! \fn hkcdShape::ShapeType hkpShape::getType() const
* The hkpCollisionDispatcher uses hkpShape types to select an appropriate hkpCollisionAgent
* for each pair of potentially colliding objects.
* If the collision dispatcher does not have a suitable registered collision agent for the objects based on their primary types, it 
* goes on to check their secondary types. The list of possible shape types is defined in hkShapeTypes. 
*/

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

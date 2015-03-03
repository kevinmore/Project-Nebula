/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpRayShapeCollectionFilter.h>

#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>
#include <Physics2012/Collide/Shape/Deprecated/CompressedMesh/hkpCompressedMeshShape.h>
#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldCollection.h>

#if !defined(HK_PLATFORM_SPU)

hkpShapeCollection::hkpShapeCollection( ShapeType type, CollectionType subType )
:	hkpShape( type )
{
	m_disableWelding = false;
	m_collectionType = subType;
}

//
//	Serialization constructor

hkpShapeCollection::hkpShapeCollection( hkFinishLoadedObjectFlag flag )
:	hkpShape(flag)
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpShapeCollection));
		m_collectionType = COLLECTION_USER;
	}
}

hkBool hkpShapeCollection::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcShpCollect",HK_NULL);

	hkpShapeBuffer shapeBuffer;
	results.changeLevel(1);
	hkpShapeKey bestKey = HK_INVALID_SHAPE_KEY;

	if ( !input.m_rayShapeCollectionFilter )
	{
		for (hkpShapeKey key = getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = getNextKey( key ) )
		{
			const hkpShape* childShape = getChildShape( key, shapeBuffer );
			if ( childShape->castRay( input, results ) )
			{
				bestKey = key;
			}
		}
	}
	else
	{
		for (hkpShapeKey key = getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = getNextKey( key ) )
		{
			if ( input.m_rayShapeCollectionFilter->isCollisionEnabled( input, *this, key ) )
			{
				const hkpShape* childShape = getChildShape( key, shapeBuffer );
				if ( childShape->castRay( input, results ) )
				{
					bestKey = key;
				}
			}
		}
	}
	results.changeLevel(-1);
	if( bestKey != HK_INVALID_SHAPE_KEY )
	{
		results.setKey(bestKey);
	}
	HK_TIMER_END();
	return bestKey != HK_INVALID_SHAPE_KEY;
}




void hkpShapeCollection::getAabb( const hkTransform& localToWorld, hkReal tolerance, hkAabb& out ) const
{
	HK_TIMER_BEGIN("hkpShapeCollection::getAabb",HK_NULL);
	out.setEmpty();

	hkpShapeBuffer shapeBuffer;
	const hkpShape* childShape;
	hkAabb aabb;

	for (hkpShapeKey key = getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = getNextKey( key ) )
	{
		childShape = getChildShape( key, shapeBuffer );
		childShape->getAabb( localToWorld, tolerance, aabb );
		out.m_min.setMin( out.m_min, aabb.m_min );
		out.m_max.setMax( out.m_max, aabb.m_max );
	}
	HK_TIMER_END();
}

hkReal hkpShapeCollection::getMaximumProjection( const hkVector4& direction ) const
{
	HK_TIMER_BEGIN("hkpShapeCollection::getMaximumProjection",HK_NULL);
	hkReal result = -HK_REAL_MAX;

	hkpShapeBuffer shapeBuffer;

	for (hkpShapeKey key = getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = getNextKey( key ) )
	{
		const hkpShape* childShape = getChildShape( key, shapeBuffer );
		const hkReal p = childShape->getMaximumProjection(direction );
		result = hkMath::max2( result, p );
	}
	HK_TIMER_END();
	return result;
}

void hkpShapeCollection::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
	HK_TIMER_BEGIN("rcShpCollect",HK_NULL);
	HK_ASSERT2(0x5c50f827,  cdBody.getShape() == this, "inconsistent cdBody, shapePointer is wrong" );

	hkpShapeBuffer shapeBuffer;

	if ( !input.m_rayShapeCollectionFilter )
	{
		for (hkpShapeKey key = getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = getNextKey( key ) )
		{
			const hkpShape* childShape = getChildShape( key, shapeBuffer );
			hkpCdBody childBody( &cdBody );
			childBody.setShape( childShape, key );
			childShape->castRayWithCollector( input, childBody, collector );
		}
	}
	else
	{
		for (hkpShapeKey key = getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = getNextKey( key ) )
		{
			if ( input.m_rayShapeCollectionFilter->isCollisionEnabled( input, *this, key ) )
			{
				const hkpShape* childShape = getChildShape( key, shapeBuffer );
				hkpCdBody childBody( &cdBody );
				childBody.setShape( childShape, key );
				childShape->castRayWithCollector( input, childBody, collector );
			}
		}
	}
	HK_TIMER_END();
}

#if !defined(HK_PLATFORM_SPU)

const hkpShapeContainer* hkpShapeCollection::getContainer() const
{
	return this;
}

#endif

void hkpShapeCollection::setWeldingInfo(hkpShapeKey key, hkInt16 weldingInfo)
{
	HK_ASSERT2( 0x3b082fa1, false, "Shape does not support welding.");
}

void hkpShapeCollection::initWeldingInfo( hkpWeldingUtility::WeldingType weldingType )
{
	HK_ASSERT2( 0x3b082fa1, false, "Shape does not support welding.");
}

#endif



/*! \fn const hkpShape* hkpShapeCollection::getChildShape(const hkpShapeKey& key, char*  buffer ) const;
* Note that if you create an object in the buffer passed in, its destructor will not be called. The buffer is simply
* deallocated when the shape is no longer needed. In general this does not matter. However if you are creating a
* shape that references another shape (for example a hkpTransformShape) in your implementation of getChildShape
* you should decrement the reference count of the referenced shape, to make up for the fact that the destructor
* of the transform shape will not be called (which would normally do this).
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

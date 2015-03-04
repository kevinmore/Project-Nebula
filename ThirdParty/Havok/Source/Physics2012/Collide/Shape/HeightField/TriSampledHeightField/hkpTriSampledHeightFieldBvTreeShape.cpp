/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldBvTreeShape.h>
#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldCollection.h>
#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>

#ifdef HK_PLATFORM_PS3
HK_COMPILE_TIME_ASSERT( (sizeof(hkpTriSampledHeightFieldBvTreeShape)&0xF) == 0); // needs to be a multiple of 16 to run on SPU
#endif

#if !defined(HK_PLATFORM_SPU)

hkpTriSampledHeightFieldBvTreeShape::hkpTriSampledHeightFieldBvTreeShape( const hkpTriSampledHeightFieldCollection* c,  hkBool doAabbRejection  )
:	hkpBvTreeShape( HKCD_SHAPE_TYPE_FROM_CLASS(hkpTriSampledHeightFieldBvTreeShape), BVTREE_TRISAMPLED_HEIGHTFIELD ), m_childContainer(c)
{
	m_wantAabbRejectionTest = doAabbRejection;
}

hkpTriSampledHeightFieldBvTreeShape::hkpTriSampledHeightFieldBvTreeShape( hkFinishLoadedObjectFlag flag )
:	hkpBvTreeShape(flag)
,	m_childContainer(flag)
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpTriSampledHeightFieldBvTreeShape));
		m_bvTreeType = BVTREE_TRISAMPLED_HEIGHTFIELD;
	}
}

#endif


void hkpTriSampledHeightFieldBvTreeShape::queryObb( const hkTransform& obbTransform, const hkVector4& obbExtent, hkReal tolerance, hkArray< hkpShapeKey >& hits ) const
{
	hkAabb aabb;
	hkAabbUtil::calcAabb( obbTransform, obbExtent, hkSimdReal::fromFloat(tolerance), aabb );

	queryAabb( aabb, hits );
}

inline void _addHit( hkpShapeKey key, hkpShapeKey* keys, int& currentHitIdx, int maxKeys)
{
	if ( currentHitIdx < maxKeys )
	{
		keys[currentHitIdx] = key;
		currentHitIdx++;
	}
}

hkBool HK_CALL hkpTriSampledHeightFieldBvTreeShape::getExtentsForQueryAabb(hkAabb& aabb, const hkpTriSampledHeightFieldCollection* collection, hkUint32& minX, hkUint32& maxX, hkUint32& minZ, hkUint32& maxZ)
{
	const hkpSampledHeightFieldShape* hf = collection->getHeightFieldShape();

	const hkReal radius = collection->getRadius();
	hkSimdReal radiusVec; radiusVec.setFromFloat(radius);

	aabb.m_max.setAdd(aabb.m_max,radiusVec);
	aabb.m_min.setSub(aabb.m_min,radiusVec);

	//{
	//	// The correction here makes sure that the y-value of the input AABB doesn't change
	//	// Alternately we could do some clever masking
	//	hkVector4 correction;
	//	correction.set(0.0f, HK_REAL_MAX, 0.0f);

	//	hkAabb thisAabb; hf->getAabb(hkTransform::getIdentity(), 0.0f, thisAabb);
	//	thisAabb.m_max.add4(correction);
	//	thisAabb.m_min.sub4(correction);
	//	// AABB to query equals intersection of (inputAabb, hfAabb) on the x/z axes
	//	aabb.m_min.setMax4(aabb.m_min, thisAabb.m_min);
	//	aabb.m_max.setMin4(aabb.m_max, thisAabb.m_max);
	//}

	// AABB to query equals intersection of (inputAabb, hfAabb) on the x/z axes
	{
		hkAabb thisAabb; hf->getAabb(hkTransform::getIdentity(), 0.0f, thisAabb);
		hkVector4Comparison xzMask; xzMask.set<hkVector4ComparisonMask::MASK_XZ>();

		hkVector4Comparison compareMins = aabb.m_min.less(thisAabb.m_min);
		hkVector4Comparison compareMaxs = aabb.m_max.greater(thisAabb.m_max);
		
		compareMins.setAnd(compareMins, xzMask);
		aabb.m_min.setSelect(compareMins, thisAabb.m_min, aabb.m_min ); //equivalent to setMax4, ignoring y and w

		compareMaxs.setAnd(compareMaxs, xzMask);
		aabb.m_max.setSelect(compareMaxs, thisAabb.m_max, aabb.m_max ); //equivalent to setMin4, ignoring y and w
	}

	if (aabb.isEmpty())
	{
		return false;
	}

	{
		// The results will be clipped to the integer values of these
		hkVector4 clipMin, clipMax;
		clipMin.setZero();
		clipMax.set( hkReal(hf->m_xRes-2), hkReal(0), hkReal(hf->m_zRes-2) );

		// <ce.todo> Pack into a single vector and do one convertToUint16WithClip call. Need to pack offset and scale too!
		HK_ALIGN16(hkIntUnion64 outMin);
		hkVector4Util::convertToUint16WithClip(aabb.m_min, hf->m_floatToIntOffsetFloorCorrected, hf->m_floatToIntScale, clipMin, clipMax, outMin );

		HK_ALIGN16(hkIntUnion64 outMax);
		hkVector4Util::convertToUint16WithClip(aabb.m_max, hf->m_floatToIntOffsetFloorCorrected, hf->m_floatToIntScale, clipMin, clipMax, outMax );

		bool   flipX = outMin.u16[0] < outMax.u16[0];
		minX = flipX ? outMin.u16[0] : outMax.u16[0];
		maxX = flipX ? outMax.u16[0] : outMin.u16[0];

		bool   flipZ = outMin.u16[2] < outMax.u16[2];
		minZ = flipZ ? outMin.u16[2] : outMax.u16[2];
		maxZ = flipZ ? outMax.u16[2] : outMin.u16[2];
	}

	return true;
}

hkUint32 hkpTriSampledHeightFieldBvTreeShape::queryAabbImpl( HKP_SHAPE_VIRTUAL_THIS const hkAabb& aabbIn, hkpShapeKey* hits, int maxNumKeys ) HKP_SHAPE_VIRTUAL_CONST
{
	const hkpTriSampledHeightFieldBvTreeShape* thisObj = static_cast<const hkpTriSampledHeightFieldBvTreeShape*>(HK_GET_THIS_PTR);

#if ! defined (HK_PLATFORM_SPU)
	const hkpTriSampledHeightFieldCollection* collection = thisObj->hkpTriSampledHeightFieldBvTreeShape::getShapeCollection();
#else
	hkpShapeBuffer buffer;
	const hkpTriSampledHeightFieldCollection* collection = thisObj->hkpTriSampledHeightFieldBvTreeShape::getShapeCollectionFromPpu(buffer);
#endif

	hkAabb aabb;

	// If the heightfield triangles have been extruded, we 'extrude' the input AABB in the opposite direction so we don't fail the AABB rejection test below
	aabb.m_min.setSub( aabbIn.m_min, collection->getTriangleExtrusion() );
	aabb.m_min.setMin( aabb.m_min, aabbIn.m_min );

	aabb.m_max.setSub( aabbIn.m_max, collection->getTriangleExtrusion() );
	aabb.m_max.setMax( aabb.m_max, aabbIn.m_max );

	hkUint32 minX, maxX, minZ, maxZ;
	hkBool overlaps = getExtentsForQueryAabb(aabb, collection, minX, maxX, minZ, maxZ);

	if (!overlaps)
	{
		return 0;
	}

	int currentHitIdx = 0;
	
	//
	// Write out list of keys
	//

#ifndef HK_PLATFORM_SPU
	const hkpSampledHeightFieldShape* hf = collection->getHeightFieldShape();

	if (m_wantAabbRejectionTest)
	{

		bool aboveHeightField = true;
		bool belowHeightField = true;
		hkReal aabbMin1 = aabb.m_min(1);
		hkReal aabbMax1 = aabb.m_max(1);
		hkReal hfScale1 = hf->m_intToFloatScale(1);

		for ( hkUint32 x = minX; x <= maxX; x++ )
		{
			for ( hkUint32 z = minZ; z <= maxZ; z++ )
			{
				_addHit( (x << 1) + (z << 16), hits, currentHitIdx, maxNumKeys);
				_addHit( ((x << 1) + (z << 16)) | 1, hits, currentHitIdx, maxNumKeys);

				if ( aboveHeightField ||  belowHeightField)
				{
					hkReal height = hfScale1 * hf->getHeightAt( x, z );
					if ( aabbMin1 < height )
					{
						aboveHeightField = false;
					}
					if ( aabbMax1 > height )
					{
						belowHeightField = false;
					}
				}
			}
		}

		if ( aboveHeightField ||  belowHeightField )
		{
			for ( hkUint32 x = minX; x <= maxX + 1; x++ )
			{
				hkReal height = hfScale1 * hf->getHeightAt( x, maxZ + 1 );
				if ( aabbMin1 < height )
				{
					aboveHeightField = false;
				}
				if ( aabbMax1 > height )
				{
					belowHeightField = false;
				}
			}
			for ( hkUint32 z = minZ; z <= maxZ + 1; z++ )
			{
				hkReal height = hfScale1 * hf->getHeightAt( maxX + 1, z );
				if ( aabbMin1 < height )
				{
					aboveHeightField = false;
				}
				if ( aabbMax1 > height )
				{
					belowHeightField = false;
				}
			}

		}

		if (aboveHeightField ||  belowHeightField )
		{
			return 0;
		}
	}
	else
#endif // #ifndef HK_PLATFORM_SPU
	{
		for ( hkUint32 x = minX; x <= maxX; x++ )
		{
			for ( hkUint32 z = minZ; z <= maxZ; z++ )
			{
				_addHit( (x << 1) + (z << 16), hits, currentHitIdx, maxNumKeys);
				_addHit( ((x << 1) + (z << 16)) | 1, hits, currentHitIdx, maxNumKeys);
			}
		}
	}

	return currentHitIdx;
}


void hkpTriSampledHeightFieldBvTreeShape::queryAabb( const hkAabb& aabbIn, hkArray<hkpShapeKey>& hits ) const
{
	hkAabb aabb; aabb = aabbIn;

	hkUint32 minX, maxX, minZ, maxZ;
#if ! defined (HK_PLATFORM_SPU)
	hkBool overlaps = getExtentsForQueryAabb(aabb, getShapeCollection(), minX, maxX, minZ, maxZ);
#else
	hkpShapeBuffer buffer;
	hkBool overlaps = getExtentsForQueryAabb(aabb, getShapeCollectionFromPpu(buffer), minX, maxX, minZ, maxZ);
#endif

	if (!overlaps)
	{
		return;
	}

#ifndef HK_PLATFORM_SPU
	const hkpSampledHeightFieldShape* hf = getShapeCollection()->getHeightFieldShape();

	int initialSize = hits.getSize();
	//
	// Write out list of keys
	//

	if (m_wantAabbRejectionTest)
	{

		bool aboveHeightField = true;
		bool belowHeightField = true;
		hkReal aabbMin1 = aabb.m_min(1);
		hkReal aabbMax1 = aabb.m_max(1);
		hkReal hfScale1 = hf->m_intToFloatScale(1);

		for ( hkUint32 x = minX; x <= maxX; x++ )
		{
			for ( hkUint32 z = minZ; z <= maxZ; z++ )
			{
				hits.pushBack((x << 1) + (z << 16));
				hits.pushBack( ((x << 1) + (z << 16)) | 1);

				if ( aboveHeightField ||  belowHeightField)
				{
					hkReal height = hfScale1 * hf->getHeightAt( x, z );
					if ( aabbMin1 < height )
					{
						aboveHeightField = false;
					}
					if ( aabbMax1 > height )
					{
						belowHeightField = false;
					}
				}
			}
		}

		if ( aboveHeightField ||  belowHeightField )
		{
			for ( hkUint32 x = minX; x <= maxX + 1; x++ )
			{
				hkReal height = hfScale1 * hf->getHeightAt( x, maxZ + 1 );
				if ( aabbMin1 < height )
				{
					aboveHeightField = false;
				}
				if ( aabbMax1 > height )
				{
					belowHeightField = false;
				}
			}
			for ( hkUint32 z = minZ; z <= maxZ + 1; z++ )
			{
				hkReal height = hfScale1 * hf->getHeightAt( maxX + 1, z );
				if ( aabbMin1 < height )
				{
					aboveHeightField = false;
				}
				if ( aabbMax1 > height )
				{
					belowHeightField = false;
				}
			}

		}

		if (aboveHeightField ||  belowHeightField )
		{
			hits.setSize( initialSize );
		}
	}
	else
#endif //#ifndef HK_PLATFORM_SPU
	{
		for ( hkUint32 x = minX; x <= maxX; x++ )
		{
			for ( hkUint32 z = minZ; z <= maxZ; z++ )
			{
				hits.pushBack((x << 1) + (z << 16));
				hits.pushBack( ((x << 1) + (z << 16)) | 1);
			}
		}
	}
}


void hkpTriSampledHeightFieldBvTreeShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
#if ! defined (HK_PLATFORM_SPU)
	getShapeCollection()->getAabb( localToWorld, tolerance, out );
#else
	hkpShapeBuffer buffer;
	getShapeCollectionFromPpu(buffer)->getAabb( localToWorld, tolerance, out );
#endif
}

hkBool hkpTriSampledHeightFieldBvTreeShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& output) const
{
#if ! defined (HK_PLATFORM_SPU)
	return getShapeCollection()->getHeightFieldShape()->castRay( input, output );
#else
	hkpShapeBuffer buffer;
	return getShapeCollectionFromPpu(buffer)->getHeightFieldShape()->castRay( input, output );
#endif
}
void hkpTriSampledHeightFieldBvTreeShape::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
#if ! defined (HK_PLATFORM_SPU)
	getShapeCollection()->getHeightFieldShape()->castRayWithCollector( input, cdBody, collector );
#else
	hkpShapeBuffer buffer;
	getShapeCollectionFromPpu(buffer)->getHeightFieldShape()->castRayWithCollector( input, cdBody, collector );
#endif
}


#if !defined(HK_PLATFORM_SPU)

int hkpTriSampledHeightFieldBvTreeShape::calcSizeForSpu(const CalcSizeForSpuInput& input, int spuBufferSizeLeft) const
{
	int childSize = m_childContainer.getChild()->calcSizeForSpu(input, HK_SPU_AGENT_SECTOR_JOB_MAX_SHAPE_SIZE);
	if( childSize < 0 || childSize > HK_SPU_AGENT_SECTOR_JOB_MAX_SHAPE_SIZE )
	{
		// early out if cascade will not fit into spu's shape buffer
		return -1;
	}

	// the spu will need this value to properly dma the child shape in one go
	m_childSize = childSize;

	// if child is not consecutive in memory, restart size calculation with just us
	return sizeof(*this);
}

#else

void hkpTriSampledHeightFieldBvTreeShape::getChildShapeFromPpu(hkpShapeBuffer& buffer) const 
{
	const hkpShape* shapeOnPpu = m_childContainer.getChild();
	int shapeOnPpuSize = m_childSize;

	const hkpShape* shapeOnSpu = reinterpret_cast<const hkpShape*>(g_SpuCollideUntypedCache->getFromMainMemory(shapeOnPpu, shapeOnPpuSize));
	HKP_PATCH_CONST_SHAPE_VTABLE( shapeOnSpu );

	// COPY over to buffer (instead of dmaing to buffer above, since we are returning this data)
	hkString::memCpy16NonEmpty( buffer, shapeOnSpu, ((shapeOnPpuSize+15)>>4) );
}

#endif

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

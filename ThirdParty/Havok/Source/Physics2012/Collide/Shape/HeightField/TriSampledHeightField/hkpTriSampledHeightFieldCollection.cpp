/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldCollection.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>

// Need these for computing welding information
#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>
#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldBvTreeShape.h>


#if !defined(HK_PLATFORM_SPU)

hkpTriSampledHeightFieldCollection::hkpTriSampledHeightFieldCollection( const hkpSampledHeightFieldShape* shape, hkReal radius )
: hkpShapeCollection( HKCD_SHAPE_TYPE_FROM_CLASS(hkpTriSampledHeightFieldCollection), COLLECTION_TRISAMPLED_HEIGHTFIELD )
{
	HK_ASSERT2(0xf89724ab, shape != HK_NULL, "You must pass a non-NULL shape pointer to this function");
	m_heightfield = shape;
	m_radius = radius;
	m_triangleExtrusion.setZero();
	m_heightfield->addReference();
	HK_ASSERT2( 0x128376ab, shape->m_xRes < 0x7fff, "X Resolution of the HeightField must be less than 32767" );
	HK_ASSERT2( 0x128377ab, shape->m_zRes < 0xffff, "Y Resolution of the HeightField must be less than 65535" );
}

//
//	Serialization constructor

hkpTriSampledHeightFieldCollection::hkpTriSampledHeightFieldCollection( hkFinishLoadedObjectFlag flag )
:	hkpShapeCollection( flag )
,	m_weldingInfo(flag)
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpTriSampledHeightFieldCollection));
		m_collectionType = COLLECTION_TRISAMPLED_HEIGHTFIELD;
	}
}

hkpTriSampledHeightFieldCollection::~hkpTriSampledHeightFieldCollection()
{
	m_heightfield->removeReference();
}

#endif

hkpShapeKey hkpTriSampledHeightFieldCollection::getFirstKey() const
{
	return 0;
}

hkpShapeKey hkpTriSampledHeightFieldCollection::getNextKey( hkpShapeKey oldKey ) const
{
	if (( oldKey & 1) == 0)
	{
		return oldKey | 1;
	}

	int x = getXFromShapeKey(oldKey);
	int z = getZFromShapeKey(oldKey);

	x += 1;
	if (x == m_heightfield->m_xRes - 1)
	{
		x = 0;
		z += 1;
		if (z == m_heightfield->m_zRes - 1)
		{
			return HK_INVALID_SHAPE_KEY;
		}
	}

	return (x << 1) + (z << 16);
}


const hkpShape* hkpTriSampledHeightFieldCollection::getChildShape(hkpShapeKey key, hkpShapeBuffer& buffer) const
{
	const hkpSampledHeightFieldShape* hf = getHeightFieldShape();
	
	const int x = getXFromShapeKey(key);
	const int z = getZFromShapeKey(key);
	
#ifdef HK_PLATFORM_SPU
	hkpTriangleShape* triangle = (hkpTriangleShape*)&buffer;
	triangle->setType( HKCD_SHAPE_TYPE_FROM_CLASS(hkpTriangleShape) );
	HKCD_PATCH_SHAPE_VTABLE( triangle );
	triangle->setRadius( m_radius );
#else
	hkpTriangleShape* triangle = new(buffer)hkpTriangleShape( m_radius ); 
#endif
	
	const hkVector4& scale = hf->m_intToFloatScale;
	
	// Doing an extra calculation here, but keeps code size down
	hkVector4 p[4];
	for (int i=0; i<4; i++)
	{
		const int tempX = x + (i >> 1);
		const int tempZ = z + (i &  1);
		p[i].set(hkReal(tempX), hf->getHeightAt(tempX, tempZ), hkReal(tempZ));
		p[i].mul(scale);
	}

	const int flipBit = (int) hf->getTriangleFlip();
	const int keyBit = key & 1;

	const int idx0 = 3*((~flipBit)&keyBit);
	const int idx1 = 1 + (keyBit * (1+flipBit));
	const int idx2 = 2 - keyBit + flipBit;

	triangle->setVertex<0>(p[idx0]);
	triangle->setVertex<1>(p[idx1]);
	triangle->setVertex<2>(p[idx2]);

	const hkpWeldingUtility::WeldingType weldingType = (m_weldingInfo.getSize() == 0) ? hkpWeldingUtility::WELDING_TYPE_NONE : hkpWeldingUtility::WELDING_TYPE_ANTICLOCKWISE;
	const hkUint16 weldingInfo = (m_weldingInfo.getSize() == 0) ? 0 : getWeldingInfo(key);
	triangle->setWeldingType(weldingType);
	triangle->setWeldingInfo(weldingInfo);

	triangle->setExtrusion(getTriangleExtrusion());

	return triangle;
}


void hkpTriSampledHeightFieldCollection::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	hkReal modifiedTolerance = tolerance + m_radius;

	const hkpSampledHeightFieldShape* heightfield = getHeightFieldShape();
	
	hkAabb tempAabb;
	heightfield->getAabb( localToWorld, modifiedTolerance, tempAabb );

	hkVector4 extrudedMin, extrudedMax;
	extrudedMin.setAdd(tempAabb.m_min, m_triangleExtrusion);
	extrudedMax.setAdd(tempAabb.m_max, m_triangleExtrusion);

	out.m_min.setMin(tempAabb.m_min, extrudedMin);
	out.m_max.setMax(tempAabb.m_max, extrudedMax);
}

#if !defined(HK_PLATFORM_SPU)

int hkpTriSampledHeightFieldCollection::calcSizeForSpu(const CalcSizeForSpuInput& input, int spuBufferSizeLeft) const
{
	// only cascades that will fit in total into one of the spu's shape buffers are allowed to be uploaded onto spu.
	int maxAvailableBufferSize = spuBufferSizeLeft - sizeof(*this);

	int childSize = m_heightfield->calcSizeForSpu(input, maxAvailableBufferSize);
	if ( childSize < 0 || childSize > maxAvailableBufferSize )
	{
		// early out if cascade will not fit into spu's shape buffer
		return -1;
	}

	// if child is consecutive in memory, set flag and return total size
	if ( hkUlong(m_heightfield) == hkUlong((this+1)) )
	{
		m_childSize = 0;
		return sizeof(*this) + childSize;
	}

	// the SPU will need this value to properly DMA the child shape in one go
	m_childSize = childSize;

	// if child is not consecutive in memory, restart size calculation with just us
	return HK_NEXT_MULTIPLE_OF( 16, sizeof(*this) );
}

#endif

hkUint32 hkpTriSampledHeightFieldCollection::getCollisionFilterInfo(hkpShapeKey key) const
{
	return 0;
}

void hkpTriSampledHeightFieldCollection::setWeldingInfo(hkpShapeKey key, hkInt16 weldingInfo)
{
	int index = getIndexFromShapeKey(key);
	HK_ASSERT3(0x3b082fa1, index >= 0 && index < m_weldingInfo.getSize(), "hkpTriSampledHeightFieldCollection does not have a triangle at index" << index);
	m_weldingInfo[index] = weldingInfo;
}


void hkpTriSampledHeightFieldCollection::initWeldingInfo( hkpWeldingUtility::WeldingType weldingType )
{
	if (weldingType != hkpWeldingUtility::WELDING_TYPE_NONE )
	{
		HK_ASSERT2(0x775dbc32, weldingType == hkpWeldingUtility::WELDING_TYPE_ANTICLOCKWISE, "Welding type for hkpTriSampledHeightFieldCollection must be anit-clockwise.");
		const hkpSampledHeightFieldShape* hf = getHeightFieldShape();
		const int numTris = 2*(hf->m_xRes - 1)*(hf->m_zRes - 1);
		m_weldingInfo.reserveExactly(numTris);
		m_weldingInfo.setSize(numTris, 0);
	}
	else
	{
		m_weldingInfo.clearAndDeallocate();
	}
}


const hkpSampledHeightFieldShape* hkpTriSampledHeightFieldCollection::getHeightFieldShape() const
{
#ifndef HK_PLATFORM_SPU
	return m_heightfield;
#else
	
	// pointer to the memory right after this shape
	hkpShape* dstInhkpShapeBuffer = const_cast<hkpShape*>( reinterpret_cast<const hkpShape*>( this+1 ) );
	dstInhkpShapeBuffer = (hkpShape*)HK_NEXT_MULTIPLE_OF( 16, (hkUlong)dstInhkpShapeBuffer );
	
	// fetch child shape if it is not yet present in SPU's local shape buffer
	if ( m_childSize > 0 )
	{
		// get child shape from main memory; put it right after this shape
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(dstInhkpShapeBuffer, m_heightfield, m_childSize, hkSpuDmaManager::READ_COPY);
		HK_SPU_DMA_PERFORM_FINAL_CHECKS(m_heightfield, dstInhkpShapeBuffer, m_childSize);

		// flag this shape as locally available
		m_childSize = 0;
	}
	
	HKCD_PATCH_SHAPE_VTABLE(dstInhkpShapeBuffer);
	return (hkpSampledHeightFieldShape*)dstInhkpShapeBuffer;
#endif
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

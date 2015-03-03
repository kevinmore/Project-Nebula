/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Composite/HeightField/Compressed/hknpCompressedHeightFieldShape.h>


#if !defined(HK_PLATFORM_SPU)

hknpCompressedHeightFieldShape::hknpCompressedHeightFieldShape(
	const hknpHeightFieldShapeCinfo& cinfo, hknpHeightFieldShape* hf )
:	hknpHeightFieldShape(cinfo),
	m_triangleFlip(false)
{
	m_storage.setSize( (m_intSizeX+1) * (m_intSizeZ+1) );

	// First pass through, compute the min and max
	hkReal min=HK_REAL_MAX, max=-HK_REAL_MAX;
	bool nonTrivialShapeTag = false;
	for(int z = 0; z < m_intSizeZ+1; z+=1)
	{
		for(int x = 0; x < m_intSizeX+1; x+=1)
		{
			hkVector4 heights; hknpShapeTag shapeTag; hkBool32 triangleFlipOut;
			hf->getQuadInfoAt( x, z, &heights, &shapeTag, &triangleFlipOut );
			hkReal height = heights.getComponent<0>().getReal();
			min = hkMath::min2(height, min);
			max = hkMath::max2(height, max);
			if (shapeTag!=HKNP_INVALID_SHAPE_TAG) nonTrivialShapeTag = true;
		}
	}

	m_offset = min;
	m_scale = (max - min) / hkReal (hkUint16(-1));

	if (nonTrivialShapeTag)
	{
		m_shapeTags.setSize( (m_intSizeX+1) * (m_intSizeZ+1) );
	}

	for(int z = 0; z < m_intSizeZ+1; z++)
	{
		for(int x = 0; x < m_intSizeX+1; x++)
		{
			hkVector4 heights; hknpShapeTag shapeTag; hkBool32 triangleFlipOut;
			hf->getQuadInfoAt( x, z, &heights, &shapeTag, &triangleFlipOut );
			hkReal height = heights.getComponent<0>().getReal();
			m_storage[z*(m_intSizeX+1) + x] = compress( height );
			if (nonTrivialShapeTag) m_shapeTags[z*(m_intSizeX+1) + x] = shapeTag;
		}
	}
}

hknpCompressedHeightFieldShape::hknpCompressedHeightFieldShape(
	const hknpHeightFieldShapeCinfo& cinfo,
	const hkArray<hkUint16>& samples, hkReal quantizationOffset, hkReal quantizationScale,
	hkArray<hknpShapeTag>* shapeTags )
:	hknpHeightFieldShape(cinfo)
,	m_triangleFlip(false)
{
	m_storage = samples;
	m_scale = quantizationScale;
	m_offset = quantizationOffset;
	if (shapeTags)
	{
		m_shapeTags = *shapeTags;
	}
}

hknpCompressedHeightFieldShape::hknpCompressedHeightFieldShape(
	const hknpHeightFieldShapeCinfo& cinfo,
	const hkArray<hkReal>& samples, hkArray<hknpShapeTag>* shapeTags )
:	hknpHeightFieldShape(cinfo)
,	m_triangleFlip(false)
{
	hkReal min=HK_REAL_MAX, max=-HK_REAL_MAX;
	for (int i=0, ei=(m_intSizeZ+1)*(m_intSizeX+1); i<ei; i++)
	{
		min = hkMath::min2(samples[i], min);
		max = hkMath::max2(samples[i], max);
	}

	m_offset = min;
	m_scale = (max - min) / hkReal (hkUint16(-1));

	m_storage.setSize( (m_intSizeX+1) * (m_intSizeZ+1) );

	for (int i=0, ei=(m_intSizeZ+1)*(m_intSizeX+1); i<ei; i++)
	{
		m_storage[i] = compress( samples[i] );
	}
}

hknpCompressedHeightFieldShape::hknpCompressedHeightFieldShape( hkFinishLoadedObjectFlag f ) :
	hknpHeightFieldShape(f),
	m_storage(f),
	m_shapeTags(f)
{

}

#endif	// !HK_PLATFORM_SPU

int hknpCompressedHeightFieldShape::calcSize() const
{
	return sizeof(hknpCompressedHeightFieldShape);
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

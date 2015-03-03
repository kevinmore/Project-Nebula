/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>
#include <Common/Base/Container/Array/hkFixedCapacityArray.h>


#if !defined(HK_PLATFORM_SPU)

hknpCompositeShape::hknpCompositeShape()
:	hknpShape( hknpCollisionDispatchType::COMPOSITE ),
	m_shapeTagCodecInfo( HKNP_INVALID_SHAPE_TAG_CODEC_INFO )
{
	m_flags.orWith( IS_COMPOSITE_SHAPE );
}

//
//	hkReferencedObject implementation

const hkClass* hknpCompositeShape::getClassType() const
{
	return &hknpCompositeShapeClass;
}

#endif // defined(HK_PLATFORM_SPU)

#if !defined(HK_PLATFORM_SPU)
void hknpCompositeShape::getAllShapeKeys( const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask, hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#else
void hknpCompositeShape::getAllShapeKeys( const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask, hkUint8* shapeBuffer, int shapeBufferSize, hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#endif
{
#if !defined(HK_PLATFORM_SPU)
	for( hkRefPtr<hknpShapeKeyIterator> it = createShapeKeyIterator(mask); it->isValid(); it->next() )	
#else
	hkUint8 iteratorBuffer[ HKNP_MAX_SHAPE_KEY_ITERATOR_SIZE_ON_SPU ];
	for( hknpShapeKeyIterator* it = createShapeKeyIterator( iteratorBuffer, HKNP_MAX_SHAPE_KEY_ITERATOR_SIZE_ON_SPU, mask ); it->isValid(); it->next() )
#endif
	{
		int subTreeKeyLength = it->getKeyPath().getKeySize();

		hknpShapeKey fullKey = shapeKeyPath.makeKey(it->getKeyPath().getKey() >> HKNP_NUM_UNUSED_SHAPE_KEY_BITS(subTreeKeyLength), subTreeKeyLength);

		hknpShapeKeyPath fullKeyPath(fullKey, shapeKeyPath.getKeySize() + subTreeKeyLength);
		keyPathsOut->pushBack(fullKeyPath);
	}
}

void hknpCompositeShape::buildEdgeWeldingMap( const hknpCompositeShape::EdgeWeld* entries, int numEntries )
{
	if (!numEntries) return;

	hkArray<hknpSparseCompactMapUtil::Entry > remapedEntries;
	remapedEntries.reserve(numEntries);
	for (int i=0; i<numEntries; i++)
	{
		hknpSparseCompactMapUtil::Entry& entry = remapedEntries.expandOne();
		entry.m_key = entries[i].m_shapeKey >> (sizeof(hknpShapeKey)*8 - m_numShapeKeyBits);
		entry.m_value = entries[i].m_edges;
	}

	int valueBits = 4;
	int keyBits = m_numShapeKeyBits;
	int primaryKeyBits = hkMath::max2(keyBits-5,2); // he: What is a reasonable default?
	m_edgeWeldingMap.buildMap(keyBits, primaryKeyBits, valueBits, &remapedEntries[0], numEntries);
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

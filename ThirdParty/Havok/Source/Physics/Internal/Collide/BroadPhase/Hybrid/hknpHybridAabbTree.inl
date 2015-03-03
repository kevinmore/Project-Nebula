/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Internal/Algorithms/Distance/hkcdDistanceAabbAabb.h>

#if defined (HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#include <Common/Base/Spu/Dma/Buffer/hkDmaBuffer.h>
#endif

template<typename KeyType, typename hkIdxType>
void hknpHybridAabbTree<KeyType, hkIdxType>::LeafData::addFooter()
{
	hkUint16 maxValue = hkUint16(-1);
	if ( NUM_SWEEP_PADDING >0 ) m_aabbs[m_numAabbs + 0].m_min[0] = maxValue;
	if ( NUM_SWEEP_PADDING >1 ) m_aabbs[m_numAabbs + 1].m_min[0] = maxValue;
	if ( NUM_SWEEP_PADDING >2 ) m_aabbs[m_numAabbs + 2].m_min[0] = maxValue;
	if ( NUM_SWEEP_PADDING >3 ) m_aabbs[m_numAabbs + 3].m_min[0] = maxValue;
}

template<typename KeyType, typename hkIdxType>
void hknpHybridAabbTree<KeyType, hkIdxType>::LeafData::removeAtAndCopy( int index )
{
	m_numAabbs--;
	for ( int i = index; i < m_numAabbs; i++) { m_aabbs[i] = m_aabbs[i+1] ;}
	m_aabbs[m_numAabbs].m_min[0] = hkUint16(-1);
}


template<typename KeyType, typename hkIdxType>
void hknpHybridAabbTree<KeyType, hkIdxType>::LeafData::calcLeafAabb(hkAabb16& aabbOut) const
{
	aabbOut.setExtents( m_aabbs, m_numAabbs );
}

template<typename KeyType, typename hkIdxType>
void hknpHybridAabbTree<KeyType, hkIdxType>::LeafData::quickSortAabbs()
{
	hkAlgorithm::quickSort( m_aabbs, m_numAabbs );
}


template<typename KeyType, typename hkIdxType>
hkIdxType hknpHybridAabbTree<KeyType, hkIdxType>::allocateLeafDataIdx()
{
	//	shouldnt expand only 1 when it runs out of the memory
	m_numLeaves ++;
	if( m_freeLeafDataIdx.getSize() )
	{
		hkIdxType freeIdx = m_freeLeafDataIdx.back(); m_freeLeafDataIdx.popBack();
		m_leafDatas[ freeIdx ].clear();
		return freeIdx;
	}
	else
	{
		hkIdxType freeIdx = (hkIdxType)m_leafDatas.getSize();
		m_leafDatas.expandOne();
		m_leafDatas[ freeIdx ].clear();
		return freeIdx;
	}
}

template<typename KeyType, typename hkIdxType>
void hknpHybridAabbTree<KeyType, hkIdxType>::deallocateLeafDataIdx(hkIdxType idx)
{
	HK_ASSERT(0x299aaad4, idx <= (hkIdxType)m_leafDatas.getCapacity());
	m_freeLeafDataIdx.pushBack( idx );
	m_numLeaves --;
}


template<typename KeyType, typename hkIdxType>
template<class VISIT>
HK_FORCE_INLINE void hknpHybridAabbTree<KeyType, hkIdxType>::castRay( const hknpHybridAabbTree<KeyType, hkIdxType>* tree, hkIdxType rootNodeIdx, const hkcdRay& rayInTreeSpace, VISIT& visitor)
{
	if( tree->m_rootNodeIdx == s_invalidIdx )
	{
		return;
	}

	hkInplaceArray<hkIdxType, 128> stack;

	hkIdxType iIdx = rootNodeIdx;

	while(1)
	{
		const Node* node = tree->getNode( iIdx );

#ifdef HK_PLATFORM_SPU
		hkTypedDmaBuffer<Node> nodeBuffer(HK_DMA_WAIT, node, hkSpuDmaManager::READ_COPY);
		node = nodeBuffer.getContents();
#endif

		if( !node->isLeaf() )
		{
			hkIdxType iIdx0 = node->getChildIdx(0);
			hkIdxType iIdx1 = node->getChildIdx(1);
			const Node* node0 = tree->getNode( iIdx0 );
			const Node* node1 = tree->getNode( iIdx1 );

#ifdef HK_PLATFORM_SPU
			hkTypedDmaBuffer<Node> nodeBuffer0(HK_DMA_DONT_WAIT, node0, hkSpuDmaManager::READ_COPY);
			node0 = nodeBuffer0.getContents();
			hkTypedDmaBuffer<Node> nodeBuffer1(HK_DMA_WAIT, node1, hkSpuDmaManager::READ_COPY);
			node1 = nodeBuffer1.getContents();
#endif

			hkAabb nodeVolume0;	hkIntSpaceUtil::convertIntToFloatAabbIntSpace( node0->m_volume, nodeVolume0 );
			hkAabb nodeVolume1;	hkIntSpaceUtil::convertIntToFloatAabbIntSpace( node1->m_volume, nodeVolume1 );

			hkSimdReal fraction0 = visitor.getEarlyOutHitFraction();
			hkSimdReal fraction1 = fraction0;
			hkBool32 intersect0 = hkcdIntersectRayAabb( rayInTreeSpace, nodeVolume0, &fraction0 );
			hkBool32 intersect1 = hkcdIntersectRayAabb( rayInTreeSpace, nodeVolume1, &fraction1 );
			if ( intersect0 & intersect1 )
			{
				// both children hit, continue with the closer child
				if ( fraction0 < fraction1 )
				{
					iIdx = iIdx0;
					stack.pushBack( iIdx1 );
				}
				else
				{
					iIdx = iIdx1;
					stack.pushBack( iIdx0 );
				}
				continue;
			}
			if ( intersect0 )
			{
				iIdx = iIdx0;
				continue;
			}
			if ( intersect1 )
			{
				iIdx = iIdx1;
				continue;
			}
			// else no hit, take next element from the stack
		}
		else
		{
			const LeafData* leaf = &tree->m_leafDatas[node->m_leafData.m_leafArrayIdx];

#ifdef HK_PLATFORM_SPU
			hkTypedDmaBuffer<LeafData> leafBuffer(HK_DMA_WAIT, leaf, hkSpuDmaManager::READ_COPY);
			leaf = leafBuffer.getContents();
#endif

			//	hit leaf (cluster)
			bool continueRay = visitor.visit(*leaf);
			if (!continueRay)
			{
				return;
			}
		}
		if ( !stack.isEmpty() )
		{
			iIdx = stack.back();
			stack.popBack();
			continue;
		}
		break;
	}
}


template<typename KeyType, typename hkIdxType>
template<class VISIT>
HK_FORCE_INLINE void hknpHybridAabbTree<KeyType, hkIdxType>::aabbCast( const hknpHybridAabbTree<KeyType, hkIdxType>* tree, hkIdxType rootNodeIdx,
	const hkcdRay& rayInTreeSpace, hkVector4Parameter halfExtentInTreeSpace, VISIT& visitor)
{
	if( tree->m_rootNodeIdx == s_invalidIdx )
	{
		return;
	}

	hkInplaceArray<hkIdxType, 128> stack;

	hkIdxType iIdx = rootNodeIdx;

	while(1)
	{
		const Node* node = tree->getNode( iIdx );

#ifdef HK_PLATFORM_SPU
		hkTypedDmaBuffer<Node> nodeBuffer(HK_DMA_WAIT, node, hkSpuDmaManager::READ_COPY);
		node = nodeBuffer.getContents();
#endif

		if( !node->isLeaf() )
		{
			hkIdxType iIdx0 = node->getChildIdx(0);
			hkIdxType iIdx1 = node->getChildIdx(1);
			const Node* node0 = tree->getNode( iIdx0 );
			const Node* node1 = tree->getNode( iIdx1 );

#ifdef HK_PLATFORM_SPU
			hkTypedDmaBuffer<Node> nodeBuffer0(HK_DMA_DONT_WAIT, node0, hkSpuDmaManager::READ_COPY);
			node0 = nodeBuffer0.getContents();
			hkTypedDmaBuffer<Node> nodeBuffer1(HK_DMA_WAIT, node1, hkSpuDmaManager::READ_COPY);
			node1 = nodeBuffer1.getContents();
#endif

			hkAabb nodeVolume0;	hkIntSpaceUtil::convertIntToFloatAabbIntSpace( node0->m_volume, nodeVolume0 );
			hkAabb nodeVolume1;	hkIntSpaceUtil::convertIntToFloatAabbIntSpace( node1->m_volume, nodeVolume1 );

			nodeVolume0.m_min.sub(halfExtentInTreeSpace);
			nodeVolume0.m_max.add(halfExtentInTreeSpace);
			nodeVolume1.m_min.sub(halfExtentInTreeSpace);
			nodeVolume1.m_max.add(halfExtentInTreeSpace);

			hkSimdReal fraction0 = visitor.getEarlyOutHitFraction();
			hkSimdReal fraction1 = fraction0;
			hkBool32 intersect0 = hkcdIntersectRayAabb( rayInTreeSpace, nodeVolume0, &fraction0 );
			hkBool32 intersect1 = hkcdIntersectRayAabb( rayInTreeSpace, nodeVolume1, &fraction1 );
			if ( intersect0 & intersect1 )
			{
				// both children hit, continue with the closer child
				if ( fraction0 < fraction1 )
				{
					iIdx = iIdx0;
					stack.pushBack( iIdx1 );
				}
				else
				{
					iIdx = iIdx1;
					stack.pushBack( iIdx0 );
				}
				continue;
			}
			if ( intersect0 )
			{
				iIdx = iIdx0;
				continue;
			}
			if ( intersect1 )
			{
				iIdx = iIdx1;
				continue;
			}
			// else no hit, take next element from the stack
		}
		else
		{
			//	hit leaf (cluster)
			const LeafData* leaf = &tree->m_leafDatas[node->m_leafData.m_leafArrayIdx];

#ifdef HK_PLATFORM_SPU
			hkTypedDmaBuffer<LeafData> leafBuffer(HK_DMA_WAIT, leaf, hkSpuDmaManager::READ_COPY);
			leaf = leafBuffer.getContents();
#endif
			visitor.visit(*leaf);
		}
		if ( !stack.isEmpty() )
		{
			iIdx = stack.back();
			stack.popBack();
			continue;
		}
		break;
	}
}

template<typename KeyType, typename hkIdxType>
void HK_CALL hknpHybridAabbTree<KeyType, hkIdxType>::queryAabb( const hknpHybridAabbTree<KeyType, hkIdxType>* tree, hkIdxType rootNodeIdx,
															   const hkAabb16& queryVolume, hkArray<hkIdxType>& leafIdxOut)
{
	if( tree->m_rootNodeIdx == s_invalidIdx )
	{
		return;
	}

	hkInplaceArray<hkIdxType, 128> stack;


	hkIdxType iIdx = rootNodeIdx;

	while(1)
	{
		const Node* node = tree->getNode( iIdx );
		const hkAabb16& qVolume = node->m_volume;
		if( !qVolume.disjoint( queryVolume ) )
		{
			if( !node->isLeaf() )
			{
				iIdx = node->getChildIdx(1);
				stack.pushBack( node->getChildIdx(0) );
				continue;
			}
			//	hit
			leafIdxOut.pushBack( node->m_leafData.m_leafArrayIdx );
		}
		if ( !stack.isEmpty() )
		{
			iIdx = stack.back();
			stack.popBack();
			continue;
		}
		break;
	}
}

template<typename KeyType, typename hkIdxType>
template<class VISIT>
HK_FORCE_INLINE void hknpHybridAabbTree<KeyType, hkIdxType>::closestPoint( const hknpHybridAabbTree<KeyType, hkIdxType>* tree, hkIdxType rootNodeIdx,
	hkVector4Parameter posInWorldSpace, hkVector4Parameter halfExtentInWorldSpace, const hkIntSpaceUtil& intSpaceUtil, VISIT& visitor)
{
	if( tree->m_rootNodeIdx == s_invalidIdx )
	{
		return;
	}

	hkInplaceArray<hkIdxType, 128> stack;
	hkIdxType iIdx = rootNodeIdx;

	while(1)
	{
		const Node* node = tree->getNode( iIdx );

#ifdef HK_PLATFORM_SPU
		hkTypedDmaBuffer<Node> nodeBuffer(HK_DMA_WAIT, node, hkSpuDmaManager::READ_COPY);
		node = nodeBuffer.getContents();
#endif

		if( !node->isLeaf() )
		{
			hkIdxType iIdx0 = node->getChildIdx(0);
			hkIdxType iIdx1 = node->getChildIdx(1);
			const Node* node0 = tree->getNode( iIdx0 );
			const Node* node1 = tree->getNode( iIdx1 );
			hkAabb nodeVolume0;	intSpaceUtil.restoreAabb( node0->m_volume, nodeVolume0 );
			hkAabb nodeVolume1;	intSpaceUtil.restoreAabb( node1->m_volume, nodeVolume1 );

			hkSimdReal maxSquaredDist = visitor.maxDistSquared();

			hkSimdReal squaredDist0 = hkcdAabbAabbDistanceSquared(nodeVolume0, posInWorldSpace, halfExtentInWorldSpace);
			hkSimdReal squaredDist1 = hkcdAabbAabbDistanceSquared(nodeVolume1, posInWorldSpace, halfExtentInWorldSpace);

			hkBool32 intersect0 = squaredDist0.isLessEqual(maxSquaredDist);
			hkBool32 intersect1 = squaredDist1.isLessEqual(maxSquaredDist);
			if ( intersect0 & intersect1 )
			{
				// both children hit, continue with the closer child
				if ( squaredDist0 < squaredDist1 )
				{
					iIdx = iIdx0;
					stack.pushBack( iIdx1 );
				}
				else
				{
					iIdx = iIdx1;
					stack.pushBack( iIdx0 );
				}
				continue;
			}
			if ( intersect0 )
			{
				iIdx = iIdx0;
				continue;
			}
			if ( intersect1 )
			{
				iIdx = iIdx1;
				continue;
			}
			// else no hit, take next element from the stack
		}
		else
		{
			//	hit leaf (cluster)
			const LeafData* leaf = &tree->m_leafDatas[node->m_leafData.m_leafArrayIdx];

#ifdef HK_PLATFORM_SPU
			hkTypedDmaBuffer<LeafData> leafBuffer(HK_DMA_WAIT, leaf, hkSpuDmaManager::READ_COPY);
			leaf = leafBuffer.getContents();
#endif
			visitor.visit(*leaf);
		}
		if ( !stack.isEmpty() )
		{
			iIdx = stack.back();
			stack.popBack();
			continue;
		}
		break;
	}
}


template<typename KeyType, typename hkIdxType>
typename hknpHybridAabbTree<KeyType, hkIdxType>::Node* hknpHybridAabbTree<KeyType, hkIdxType>::NodeManager::getNode( hkIdxType idx )
{
	HK_ASSERT(0x17e484b, idx < hkIdxType(m_nodes.getSize()) );
	return &m_nodes[idx];
}

template<typename KeyType, typename hkIdxType>
const typename hknpHybridAabbTree<KeyType, hkIdxType>::Node* hknpHybridAabbTree<KeyType, hkIdxType>::NodeManager::getNode( hkIdxType idx ) const
{
	HK_ASSERT(0x1396e0f5, idx < hkIdxType(m_nodes.getSize()) );
	return &m_nodes[idx];
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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>

#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldBvTreeShape.h>

#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>
#include <Physics2012/Internal/Collide/BvCompressedMesh/hkpBvCompressedMeshShape.h>

#include <Physics2012/Collide/Shape/Query/hkpAabbCastCollector.h>



#if !defined(HK_PLATFORM_SPU)

hkpBvTreeShape::hkpBvTreeShape( hkFinishLoadedObjectFlag flag )
:	hkpShape(flag)
{ 
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpBvTreeShape));
		m_bvTreeType = BVTREE_USER;
	}
}

#else

hkpBvTreeShape::BvTreeFuncs hkpBvTreeShape::s_bvTreeFunctions[ hkpBvTreeShape::BVTREE_MAX ];


void HK_CALL hkpBvTreeShape::registerBvTreeCollideFunctions()
{
	s_bvTreeFunctions[BVTREE_MOPP].m_queryAabbFunc = &hkpMoppBvTreeShape::queryAabbImpl;
	s_bvTreeFunctions[BVTREE_TRISAMPLED_HEIGHTFIELD].m_queryAabbFunc = &hkpTriSampledHeightFieldBvTreeShape::queryAabbImpl;


	// Users may register their custom implementation here.
	// s_bvTreeFunctions[BVTREE_USER].m_queryAabbFunc = ...
}

void HK_CALL hkpBvTreeShape::registerBvTreeCollideFunctions_StaticCompound()
{
	s_bvTreeFunctions[BVTREE_MOPP].m_queryAabbFunc = &hkpMoppBvTreeShape::queryAabbImpl;
	s_bvTreeFunctions[BVTREE_TRISAMPLED_HEIGHTFIELD].m_queryAabbFunc = &hkpTriSampledHeightFieldBvTreeShape::queryAabbImpl;
	s_bvTreeFunctions[BVTREE_STATIC_COMPOUND].m_queryAabbFunc = &hkpStaticCompoundShape::queryAabbImpl;
	s_bvTreeFunctions[BVTREE_COMPRESSED_MESH].m_queryAabbFunc = &hkpBvCompressedMeshShape::queryAabbImpl;

	// Users may register their custom implementation here.
	// s_bvTreeFunctions[BVTREE_USER].m_queryAabbFunc = ...
}

void HK_CALL hkpBvTreeShape::registerBvTreeCollideQueryFunctions()
{
	s_bvTreeFunctions[BVTREE_MOPP].m_queryAabbFunc = &hkpMoppBvTreeShape::queryAabbImpl;
	s_bvTreeFunctions[BVTREE_MOPP].m_castAabbFunc = &hkpBvTreeShape::castAabbImpl;

	s_bvTreeFunctions[BVTREE_TRISAMPLED_HEIGHTFIELD].m_queryAabbFunc = &hkpTriSampledHeightFieldBvTreeShape::queryAabbImpl;	
	s_bvTreeFunctions[BVTREE_TRISAMPLED_HEIGHTFIELD].m_castAabbFunc = &hkpBvTreeShape::castAabbImpl;

	s_bvTreeFunctions[BVTREE_STATIC_COMPOUND].m_queryAabbFunc = &hkpStaticCompoundShape::queryAabbImpl;
	s_bvTreeFunctions[BVTREE_STATIC_COMPOUND].m_castAabbFunc = &hkpStaticCompoundShape::castAabbImpl;

	s_bvTreeFunctions[BVTREE_COMPRESSED_MESH].m_queryAabbFunc = &hkpBvCompressedMeshShape::queryAabbImpl;
	s_bvTreeFunctions[BVTREE_COMPRESSED_MESH].m_castAabbFunc = &hkpBvCompressedMeshShape::castAabbImpl;
	
	// Register custom implementation functions here
	// s_bvTreeFunctions[BVTREE_USER].m_queryAabbFunc = ...
}

HK_SHAPE_CONTAINER* hkpBvTreeShape::getContainerImpl(const hkpShape* shape, hkpShapeBuffer& buffer)
{
	switch( shape->getType() )
	{
		case hkcdShapeType::MOPP:
		{
			const hkpMoppBvTreeShape* bv = static_cast<const hkpMoppBvTreeShape*>(shape);
			return (HK_SHAPE_CONTAINER*)bv->hkpMoppBvTreeShape::getShapeCollectionFromPpu(buffer);
		}

		case hkcdShapeType::BV_TREE:  
		{ 	 
			const hkpBvTreeShape* bv = static_cast<const hkpBvTreeShape*> (shape);
			switch (bv->m_bvTreeType)
			{
				case BVTREE_TRISAMPLED_HEIGHTFIELD:
				case BVTREE_MOPP:
				case BVTREE_STATIC_COMPOUND:
				case BVTREE_COMPRESSED_MESH:
				{
					HK_ASSERT2(0x401c81a0, false, "hkpBvTreeShape has inconsistent shape type and bvtree type.");
					break;
				}

				case BVTREE_USER:
				{
					// It's possible to include your own bounding-volume shape on the SPU by calling its methods from here (experts only).
					HK_WARN_ONCE(0x2836fa26, "User hkpBvTreeShape encountered on SPU");
					break;
				}

				default:
				{
					HK_ASSERT2(0x6fcaee44, 0, "Invalid hkpBvTreeShape encountered on SPU");
					break;
				}
			}
		}

		case hkcdShapeType::LIST:
		{
			const hkpListShape* list = static_cast<const hkpListShape*>( shape );
			return (HK_SHAPE_CONTAINER*) list;
		}

		case hkcdShapeType::STATIC_COMPOUND:
		{
			const hkpStaticCompoundShape* compound = static_cast<const hkpStaticCompoundShape*>( shape );
			return (HK_SHAPE_CONTAINER*) compound;
		}

		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
		{
			const hkpTriSampledHeightFieldBvTreeShape* hf = static_cast<const hkpTriSampledHeightFieldBvTreeShape*>( shape );
			return (HK_SHAPE_CONTAINER*)hf->hkpTriSampledHeightFieldBvTreeShape::getShapeCollectionFromPpu(buffer); 
		}

		case hkcdShapeType::BV_COMPRESSED_MESH:
		{
			const hkpBvCompressedMeshShape* cms = static_cast<const hkpBvCompressedMeshShape*>( shape );
			return (HK_SHAPE_CONTAINER*) cms;
		}

		default:
		{
			HK_ERROR( 0xad8755ab, "This hkpBvTreeShape type is not supported on the SPU" );
			return HK_NULL;
		}
	}
}

#endif

void hkpBvTreeShape::castAabbImpl(HKP_SHAPE_VIRTUAL_THIS const hkAabb& from, hkVector4Parameter to, hkpAabbCastCollector& collector) HKP_SHAPE_VIRTUAL_CONST
{
	const hkpBvTreeShape* bvTree = static_cast<const hkpBvTreeShape*> (HK_GET_THIS_PTR);

	// Calculate query aabb enclosing the original one in the initial and final positions
	hkAabb aabb = from;
	{
		hkVector4 center; from.getCenter(center);
		hkVector4 path; path.setSub(to, center);
		hkVector4 pathNeg; pathNeg.setMin(path, hkVector4::getZero());
		hkVector4 pathPos; pathPos.setMax(path, hkVector4::getZero());
		aabb.m_min.add(pathNeg);
		aabb.m_max.add(pathPos);		
	}

	// Run the query	
	hkpShapeKey* keys = hkAllocateStack<hkpShapeKey>(HK_MAX_NUM_HITS_PER_AABB_QUERY, "ShapeKeys");
	HK_SPU_STACK_POINTER_CHECK();
	int numKeys = bvTree->queryAabb(aabb, keys, HK_MAX_NUM_HITS_PER_AABB_QUERY);

	// Check number of results and shrink stack usage if possible
	HK_WARN_ON_DEBUG_IF(numKeys > HK_MAX_NUM_HITS_PER_AABB_QUERY, 0x16a02337, "queryAabb produced too many results (" << numKeys << ")");	
	numKeys = hkMath::_min2<int>(numKeys, HK_MAX_NUM_HITS_PER_AABB_QUERY);
	if (numKeys == 0)
	{
		hkDeallocateStack(keys, HK_MAX_NUM_HITS_PER_AABB_QUERY);
		return;
	}
	int keysSize = ((numKeys < HK_MAX_NUM_HITS_PER_AABB_QUERY) && hkShrinkAllocatedStack(keys, numKeys) ? numKeys : HK_MAX_NUM_HITS_PER_AABB_QUERY);

	// Add results to collector
	for (int i = 0; i < numKeys; ++i)
	{
		collector.addHit(keys[i]);
	}

	hkDeallocateStack(keys, keysSize);
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

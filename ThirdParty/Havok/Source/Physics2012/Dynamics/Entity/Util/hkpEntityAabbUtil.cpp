/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#	include <Physics2012/Collide/Query/Multithreaded/Spu/hkpSpuConfig.h>
#endif

#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Shape/hkpShapeContainer.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>
#include <Physics2012/Dynamics/Entity/hkpEntity.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.h>

void hkpEntityAabbUtil::entityBatchRecalcAabb(const hkpCollisionInput* collisionInput, hkpEntity*const* entityBatch, int numEntities)
{
	//HK_TIMER_BEGIN("entityBatchRecalcAabb",this);
	const hkReal tolerance = collisionInput->getTolerance() * 0.5f;

	

	for (int entityIndex = 0; entityIndex < numEntities; entityIndex++)
	{
		hkpEntity* entity = entityBatch[entityIndex];

		HK_ON_CPU( hkpCollidable*   collidable = const_cast<hkpCollidable*>  ( entity->getCollidable() ) ); //this should be a getCollidableRw() but that'll assert in multithreaded fulldebug mode; the const cast is allowed as we are only modifying the cached AABBs, which is thread safe
		HK_ON_SPU( hkCollidablePpu* collidable = const_cast<hkCollidablePpu*>( entity->getCollidable() ) );  

		const hkpShape* shape = collidable->m_shape;

		const bool isCompound = shape->getType() == hkcdShapeType::STATIC_COMPOUND;
		const int capacityChildShapeAabbs = collidable->m_boundingVolumeData.m_capacityChildShapeAabbs;

		if ( !isCompound && capacityChildShapeAabbs ) // must be run even after the last child shape is disabled, or invalidate the AABB's when numChildShapeAabbs is zero
		{
			//
			// Get a pointer to the buffer for the cached AABBs for all children. This buffer is going to get filled next.
			//

#if !defined(HK_PLATFORM_SPU)
			const hkpShapeContainer* container = shape->getContainer();
			const int numChildShapeAabbs = HK_ACCESS_COLLECTION_METHOD(container, getNumChildShapes());
			//collidable->m_boundingVolumeData.m_numChildShapeAabbs = hkUint16(numChildShapeAabbs);	// done later

			hkAabbUint32* childShapeAabbs = HK_NULL;
			hkpShapeKey*  childShapeKeys  = HK_NULL;

			hkLocalBuffer<hkAabbUint32> aabbBuffer(numChildShapeAabbs); 
			hkLocalBuffer<hkpShapeKey> keyBuffer(numChildShapeAabbs);

			//if (numChildShapeAabbs)
			//{
				childShapeAabbs = aabbBuffer.begin();
				childShapeKeys  = keyBuffer.begin();

				hkAabbUtil::OffsetAabbInput input; 
				hkAabbUtil::initOffsetAabbInput(entity->m_motion.getMotionState(), input);

				//
				// Calculate the AABBs of all children.
				// Also calculate the root AABBs (swept and unswept) from the child AABBs.
				//
				
				{
					hkpShapeBuffer shapeBuffer;										
					hkAabb rootAabb; rootAabb.setEmpty();
					hkAabb rootSweptAabb = rootAabb;
					hkAabbUint32* da = childShapeAabbs;
					hkpShapeKey* dk = childShapeKeys;

					// Note: The AABB's order MUST BE THE SAME as the order of the keys.
					for (hkpShapeKey key = HK_ACCESS_COLLECTION_METHOD(container, getFirstKey()); key != HK_INVALID_SHAPE_KEY; key = HK_ACCESS_COLLECTION_METHOD(container, getNextKey(key)))
					{

						HK_ON_CPU( const hkpShape* child = container->getChildShape(key, shapeBuffer) );
						HK_ON_SPU( const hkpShape* child = container->getChildShape(container, key, shapeBuffer) );

						// Get AABBs.
						hkAabb aabb;		child->getAabb(entity->m_motion.getTransform(), tolerance, aabb);
						hkAabb sweptAabb;	hkAabbUtil::sweepOffsetAabb(input, aabb, sweptAabb);

						// Add to the root-collidable AABBs. 
						rootAabb.m_min.setMin(rootAabb.m_min, aabb.m_min);
						rootAabb.m_max.setMax(rootAabb.m_max, aabb.m_max);
						rootSweptAabb.m_min.setMin(rootSweptAabb.m_min, sweptAabb.m_min);
						rootSweptAabb.m_max.setMax(rootSweptAabb.m_max, sweptAabb.m_max);						

						// Convert to integer space and compress
						hkAabbUtil::convertAabbToUint32(aabb,      collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, *da);

						hkAabbUint32 sweptAabbUint32;
						hkAabbUtil::convertAabbToUint32(sweptAabb, collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, sweptAabbUint32 );
						hkAabbUtil::compressExpandedAabbUint32(sweptAabbUint32, *da);

						da->m_shapeKeyByte = hkUint8(key); // just always put it there.

						*dk = key;

						da++;
						dk++;
					}
					collidable->m_boundingVolumeData.m_numChildShapeAabbs = hkUint16(dk - childShapeKeys); // done later

					// If all children are disabled the center of mass is used to obtain a valid root AABB
					const hkVector4& centerOfMass = entity->getMotion()->getCenterOfMassInWorld();
					hkVector4Comparison childrenEnabled = rootAabb.m_max.greaterEqual(rootAabb.m_min);
					rootAabb.m_max.setSelect(childrenEnabled, rootAabb.m_max, centerOfMass);
					rootAabb.m_min.setSelect(childrenEnabled, rootAabb.m_min, centerOfMass);
					rootSweptAabb.m_max.setSelect(childrenEnabled, rootSweptAabb.m_max, centerOfMass);
					rootSweptAabb.m_min.setSelect(childrenEnabled, rootSweptAabb.m_min, centerOfMass);					

					// Convert root-collidable AABB to integer space too.
#ifdef HK_ARCH_ARM
					HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0x3) == 0, "Unaligned bounding volume data!");
#else
					HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0xF) == 0, "Unaligned bounding volume data!");
#endif
					hkAabbUint32& rootAabbUint32 = reinterpret_cast<hkAabbUint32&>(collidable->m_boundingVolumeData);
					hkAabbUint32 rootSweptAabbUint32;					
					hkAabbUtil::convertAabbToUint32(rootAabb,      collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, rootAabbUint32     );
					hkAabbUtil::convertAabbToUint32(rootSweptAabb, collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, rootSweptAabbUint32);
					hkAabbUtil::compressExpandedAabbUint32(rootSweptAabbUint32, rootAabbUint32);
				}
#else
			hkAabbUint32* childShapeAabbs = HK_NULL;
			hkpShapeKey* childShapeKeys = HK_NULL;
			int numChildShapeAabbs = 0;

			HK_ALIGN16( hkUint8 shapeBuffer[HK_SPU_MAXIMUM_SHAPE_SIZE] );
			if ( shape->getType() == hkcdShapeType::MOPP )
			{
				const hkpMoppBvTreeShape* moppShape = static_cast<const hkpMoppBvTreeShape*>( shape );
				hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(&shapeBuffer[0], moppShape->getChild(), HK_SPU_MAXIMUM_SHAPE_SIZE, hkSpuDmaManager::READ_COPY);
				HK_SPU_DMA_PERFORM_FINAL_CHECKS(moppShape->getChild(), &shapeBuffer[0], HK_SPU_MAXIMUM_SHAPE_SIZE);
				shape = reinterpret_cast<hkpShape*>( &shapeBuffer[0] );
				HKP_PATCH_CONST_SHAPE_VTABLE( shape );
			}
			if ( shape->getType() == hkcdShapeType::LIST )
			{
				const hkpListShape* list = static_cast<const hkpListShape*>( shape );

				numChildShapeAabbs = list->hkpListShape::getNumChildShapes();
				//int numChildInfos = list->m_childInfo.getSize();

				collidable->m_boundingVolumeData.m_numChildShapeAabbs = hkUint16(numChildShapeAabbs);

				if (numChildShapeAabbs)
				{

					// The childShapeAabbs will hold childInfos and childShapeAabbs at the same time.
					// AABBs will be the result, child infos is intermediate data, that will be read & discarded from left to right.

					int maxAabbSize = list->getNumAabbsForSharedBufferForAabbsAndChildInfos();

					childShapeAabbs = hkAllocateStack<hkAabbUint32>(maxAabbSize); // See 0xaf5241e9.
					childShapeKeys = hkAllocateStack<hkpShapeKey>(HK_NEXT_MULTIPLE_OF(4, numChildShapeAabbs)); 

					hkAabbUtil::OffsetAabbInput input; 
					hkAabbUtil::initOffsetAabbInput(entity->m_motion.getMotionState(), input);

					//
					// Calculate the AABBs of all children.
					// Also calculate the root AABBs (swept and unswept) from the child AABBs.
					//
					if ( (collidable->m_forceCollideOntoPpu & hkCollidablePpu::FORCE_PPU_SHAPE_REQUEST) == 0 )
					{
						HK_ASSERT2(0x46aefcee, (((hkUlong)&collidable->m_boundingVolumeData) & 0xF) == 0, "Unaligned bounding volume data!");
						hkAabbUint32& rootAabbUint32 = reinterpret_cast<hkAabbUint32&>(collidable->m_boundingVolumeData);
						list->getAabbWithChildShapes(*collisionInput, input, entity->m_motion.getTransform(), entity->m_motion.getCenterOfMassInWorld(), tolerance, rootAabbUint32, childShapeAabbs, childShapeKeys, numChildShapeAabbs, maxAabbSize*sizeof(hkAabbUint32));
					}
					else
					{
						// this orders the broadphase job to process this shape
						collidable->m_boundingVolumeData.invalidate();
					}
				}
				else
				{
					// this is a valid case
					// set rootAabb to center of mass

					const hkVector4& massCenter = entity->m_motion.getCenterOfMassInWorld();
					hkAabb rootAabb; rootAabb.m_min = massCenter; rootAabb.m_max = massCenter;	// don't create an empty one as all children might be disabled. At least we are getting a valid AABB

					hkAabbUint32 rootSweptAabbUint32;
					hkAabbUint32& rootAabbUint32 = reinterpret_cast<hkAabbUint32&>(collidable->m_boundingVolumeData);
					hkAabbUtil::convertAabbToUint32(rootAabb, collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, rootAabbUint32     );
					hkAabbUtil::convertAabbToUint32(rootAabb, collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, rootSweptAabbUint32);
					hkAabbUtil::compressExpandedAabbUint32(rootSweptAabbUint32, rootAabbUint32);
				}
			}
			else
			{
				// this orders the broadphase job to process this shape
				collidable->m_boundingVolumeData.invalidate();
			}
#endif

			if (numChildShapeAabbs)
			{ 
				// now lets sort the child shapes
				HK_ASSERT2(0xad808141, childShapeAabbs && childShapeKeys, "Attempting to sort AABB's although we failed to extract them in the first place.");

				// scope needed to deallocate hkLocalBuffer<hkValueIndexPair> data
				{
					hkLocalBuffer<hkValueIndexPair> data( numChildShapeAabbs); // See 0xaf5241ea.
					hkValueIndexPair* d = data.begin();

					if ( numChildShapeAabbs > 1)
					{
						hkAabbUint32* a = childShapeAabbs;
						for (int i = 0; i < numChildShapeAabbs; i++){ d->m_index=i; d->m_value = a->m_min[0]; a++; d++; }
						hkSort( data.begin(), numChildShapeAabbs );
					}
					else if (numChildShapeAabbs == 1)
					{
						d[0].m_index = 0;				
					}

					{
						// Reorder AABB's & write shapeKeys
						HK_ASSERT2(0xad808142, numChildShapeAabbs <= capacityChildShapeAabbs, "Insufficient capacity to store child shape AABBs" );

#if ! defined (HK_PLATFORM_SPU)
						hkAabbUint32* dstAabbs = collidable->m_boundingVolumeData.m_childShapeAabbs;
						hkpShapeKey* dstKeys = collidable->m_boundingVolumeData.m_childShapeKeys;
						d = data.begin();
						for (int i = 0; i < numChildShapeAabbs; i++)
						{
							*(dstAabbs++) = childShapeAabbs[d->m_index];
							*(dstKeys++) = childShapeKeys[d->m_index];
							d++;
						}

						#ifdef HK_DEBUG
						for ( int i =0 ; i < numChildShapeAabbs-1; i++ )
						{
							HK_ASSERT( 0xf0341232, collidable->m_boundingVolumeData.m_childShapeAabbs[i].m_min[0] <= collidable->m_boundingVolumeData.m_childShapeAabbs[i+1].m_min[0]);
						}
						#endif

#else
						// sort in place
						hkAabbUint32* s = childShapeAabbs;
						hkpShapeKey* k = childShapeKeys;

						d = data.begin();
						for (int i=0; i < numChildShapeAabbs; s++, k++, d++, i++)
						{
							int srcIndex = d->m_index;
							if ( srcIndex == i )
							{
								continue;
							}
							HK_ASSERT(0x4f5ac245, srcIndex > i);
							hkAabbUint32 tmpAabb = *s;
							hkpShapeKey tmpKey = *k;

							int currentIdx = i;

							while(1) 
							{
								childShapeAabbs[currentIdx] = childShapeAabbs[srcIndex];
								childShapeKeys[currentIdx] = childShapeKeys[srcIndex];
								//data[currentIdx] = data[srcIndex];
								data[currentIdx].m_index = currentIdx;

								currentIdx = srcIndex;
								srcIndex = data[currentIdx].m_index;

								if (srcIndex == i)
								{
									childShapeAabbs[currentIdx] = tmpAabb;
									childShapeKeys[currentIdx] = tmpKey;
									data[currentIdx].m_index = currentIdx;
									break;
								}
							}
						}
						
						#ifdef HK_DEBUG
						for ( int i =0 ; i < numChildShapeAabbs-1; i++ )
						{
							HK_ASSERT( 0xf0341232, childShapeAabbs[i].m_min[0] <= childShapeAabbs[i+1].m_min[0]);
						}
						#endif
#endif

					}
				}


#if defined(HK_PLATFORM_SPU)
				//
				// Write the AABBs for all children back to PPU.
				//
				const int sizeOfChildShapeAabbs = numChildShapeAabbs * sizeof(hkAabbUint32);
				const int sizeOfChildShapeKeys  = HK_NEXT_MULTIPLE_OF(16, numChildShapeAabbs * sizeof(hkpShapeKey));
				// use the default STALL dma-group for both trasfers below
				hkSpuDmaManager::putToMainMemory(collidable->m_boundingVolumeData.m_childShapeAabbs, childShapeAabbs, sizeOfChildShapeAabbs, hkSpuDmaManager::WRITE_NEW);
				hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(collidable->m_boundingVolumeData.m_childShapeKeys, childShapeKeys, sizeOfChildShapeKeys, hkSpuDmaManager::WRITE_NEW);

				HK_SPU_DMA_PERFORM_FINAL_CHECKS(collidable->m_boundingVolumeData.m_childShapeAabbs, childShapeAabbs, sizeOfChildShapeAabbs);
				HK_SPU_DMA_PERFORM_FINAL_CHECKS(collidable->m_boundingVolumeData.m_childShapeKeys, childShapeKeys, sizeOfChildShapeKeys);
				HK_ASSERT2(0xad808142, childShapeAabbs && childShapeKeys, "Internal error.");
				hkDeallocateStack(childShapeKeys);
				hkDeallocateStack(childShapeAabbs);
#endif
			}
		}
		else
		{
			//
			// Simple shape (no container) 
			//
			hkAabb aabb;		shape->getAabb(entity->m_motion.getTransform(), tolerance, aabb);
			hkAabb sweptAabb;	hkAabbUtil::sweepAabb(entity->m_motion.getMotionState(), tolerance, aabb, sweptAabb);

			
			//hkAabbUint32& aabbUint32 = collidable->m_boundingVolumeData.getAabbUint32();
			#ifdef HK_ARCH_ARM
				HK_ASSERT2(0x5237ca31, (((hkUlong)&collidable->m_boundingVolumeData) & 0x3) == 0, "Unaligned bounding volume data!");
			#else
				HK_ASSERT2(0x5237ca31, (((hkUlong)&collidable->m_boundingVolumeData) & 0xF) == 0, "Unaligned bounding volume data!");
			#endif
			hkAabbUint32& aabbUint32 = reinterpret_cast<hkAabbUint32&>(collidable->m_boundingVolumeData);
			hkAabbUint32 sweptAabbUint32;

			hkAabbUtil::convertAabbToUint32(aabb,      collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, aabbUint32     );
			hkAabbUtil::convertAabbToUint32(sweptAabb, collisionInput->m_aabb32Info.m_bitOffsetLow, collisionInput->m_aabb32Info.m_bitOffsetHigh, collisionInput->m_aabb32Info.m_bitScale, sweptAabbUint32);
			hkAabbUtil::compressExpandedAabbUint32(sweptAabbUint32, aabbUint32);
		}
	}
	//HK_TIMER_END();
}

#if ! defined (HK_PLATFORM_SPU)
void hkpEntityAabbUtil::entityBatchInvalidateAabb(hkpEntity*const* entityBatch, int numEntities)
{
	for (int entityIndex = 0; entityIndex < numEntities; entityIndex++)
	{
		entityBatch[entityIndex]->getCollidableRw()->m_boundingVolumeData.invalidate();
	}
}
#endif

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

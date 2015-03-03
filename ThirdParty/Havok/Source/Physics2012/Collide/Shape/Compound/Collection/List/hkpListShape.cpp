/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Container/BitField/hkBitField.h>
#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Memory/PlatformUtils/Spu/SpuDmaCache/hkSpu4WayCache.h>
#	include <Physics2012/Collide/Filter/Spu/hkpSpuCollisionFilterUtil.h>
#endif

#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Query/hkpRayShapeCollectionFilter.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#include <Common/Base/Algorithm/Collide/1AxisSweep/hk1AxisSweep.h>

#if defined(HK_PLATFORM_SPU)
#	include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShapeGetAabbSpuPipelines.h>
#endif

#if defined(HK_PLATFORM_PS3_PPU)
HK_COMPILE_TIME_ASSERT(sizeof(hkpListShape::ChildInfo) * 2 == sizeof(hkAabbUint32));
#endif


#if !defined(HK_PLATFORM_SPU)

hkpListShape::hkpListShape(const hkpShape*const* shapeArray, int numShapes, hkpShapeContainer::ReferencePolicy ref) 
: hkpShapeCollection( HKCD_SHAPE_TYPE_FROM_CLASS(hkpListShape), COLLECTION_LIST )
{
	m_childInfo.reserve(4); // so that our array gets aligned16 to be downloaded to the spu
	
	setShapes( shapeArray, numShapes, HK_NULL, ref );
	// no need to call recalcAabbExtents explicitly since it is called in setShapes
	
	for (int i = 0; i < MAX_DISABLED_CHILDREN/32; i++)
	{ 
		m_enabledChildren[i] = unsigned(-1);
	}
	m_numDisabledChildren = 0;
	m_flags = ALL_FLAGS_CLEAR;
}

hkpListShape::hkpListShape( class hkFinishLoadedObjectFlag flag )
:	hkpShapeCollection(flag)
,	m_childInfo(flag)
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpListShape));
		m_collectionType = COLLECTION_LIST;
	}
}

#endif

#if !defined(HK_PLATFORM_SPU)
hkpListShape::~hkpListShape()
{
	for (int i = 0; i < m_childInfo.getSize(); i++)
	{
		m_childInfo[i].m_shape->removeReference();
	}
}

void hkpListShape::setShapes( const hkpShape*const* shapeArray, int numShapes, const hkUint32* filterInfo, hkpShapeContainer::ReferencePolicy ref )
{
	HK_ASSERT2(0x282822c7,  m_childInfo.getSize()==0, "You can only call setShapes once during construction.");
	HK_ASSERT2(0x221e5b17,  numShapes, "You cannot create a hkpListShape with no child shapes" );

	m_childInfo.setSize(numShapes);
	for (int i = 0; i < numShapes; i++)
	{
		if (shapeArray[i] != HK_NULL)
		{
			m_childInfo[i].m_shape = shapeArray[i];
			m_childInfo[i].m_collisionFilterInfo = filterInfo? filterInfo[i] : 0;
			m_childInfo[i].m_numChildShapes = numShapes;
			m_childInfo[i].m_shapeSize = 0;
			m_childInfo[i].m_shapeInfo = 0;
		}
	}

	if (ref == hkpShapeContainer::REFERENCE_POLICY_INCREMENT)
	{
		hkReferencedObject::addReferences(&m_childInfo[0].m_shape, m_childInfo.getSize(), sizeof(m_childInfo[0]));
	}

	recalcAabbExtents();
}
#endif

void hkpListShape::disableChild( hkpShapeKey index )
{
	HK_ASSERT2( 0xf0f34fe5, index < MAX_DISABLED_CHILDREN && int(index) < m_childInfo.getSize(), "You can only disable the first 256 children" );
	int bitPattern = ~(1<<(index&0x1f));
	int i = index>>5;
	int value = m_enabledChildren[ i ];
	int newVal = value & bitPattern;
	if ( value != newVal )
	{
		m_enabledChildren[i] = newVal;
		m_numDisabledChildren++;
	}
}

/// Allows for quickly enabling a child shape.
void hkpListShape::enableChild( hkpShapeKey index )
{
	HK_ASSERT2( 0xf0f34fe6, index < MAX_DISABLED_CHILDREN && int(index) < m_childInfo.getSize(), "You can only disable the first 256 children" );
	int bitPattern = (1<<(index&0x1f));
	int i = index>>5;
	int value = m_enabledChildren[ i ];
	int newVal = value | bitPattern;
	if ( value != newVal )
	{
		m_enabledChildren[i] = newVal;
		m_numDisabledChildren--;
	}
}

#ifndef HK_PLATFORM_SPU
void hkpListShape::setEnabledChildren( const hkBitField& enabledChildren )
{
	HK_ASSERT2( 0xf03465fe, enabledChildren.getSize() == m_childInfo.getSize(), "Your bitfield does not match the list shape" );
	HK_ASSERT2( 0xf03465fe, enabledChildren.getSize() <= 256, "Your bitfield is too large, you can only disable 256 children" );

	const hkUint32* HK_RESTRICT enabledChildrenWords = enabledChildren.getWords();
	for (int i =0; i < enabledChildren.getNumWords(); i++)
	{
		m_enabledChildren[i] = enabledChildrenWords[i];
	}
	m_numDisabledChildren = hkUint16(m_childInfo.getSize() - enabledChildren.bitCount());
}
#endif


hkBool hkpListShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcList",HK_NULL);

	const hkpShapeRayCastOutput originalResults = results;	

	hkpShapeKey bestKey = HK_INVALID_SHAPE_KEY;

	if ( !input.m_rayShapeCollectionFilter )
	{
		for (int i = 0; i < m_childInfo.getSize(); i++)
		{
			if (isChildEnabled(i))
			{
#if !defined(HK_PLATFORM_SPU)
				const hkpShape* childShape = m_childInfo[i].m_shape;
#else
				hkpShapeBuffer shapeBuffer;
				const hkpShape* childShape = getChildShape(i, shapeBuffer);
#endif
				results.setKey(i);
				results.changeLevel(1);
				if ( childShape->castRay( input, results ) )
				{
					bestKey = i;
				}
				results.changeLevel(-1);
			}
		}
	}
	else
	{
		for (int i = 0; i < m_childInfo.getSize(); i++)
		{
#if !defined(HK_PLATFORM_SPU)
			if ( isChildEnabled(i) && (false!=input.m_rayShapeCollectionFilter->isCollisionEnabled( input, *this, i )) )
#else
			if ( isChildEnabled(i) && hkpSpuCollisionFilterUtil::s_rayShapeContainerIsCollisionEnabled( input, *this, i) )
#endif
			{
#if !defined(HK_PLATFORM_SPU)
				const hkpShape* childShape = m_childInfo[i].m_shape;
#else
				hkpShapeBuffer shapeBuffer;
				const hkpShape* childShape = getChildShape(i, shapeBuffer);
#endif
				results.setKey(i);
				results.changeLevel(1);
				if ( childShape->castRay( input, results ) )
				{
					bestKey = i;
				}
				results.changeLevel(-1);
			}
		}
	}

	results.setKey(bestKey);

	if (bestKey == HK_INVALID_SHAPE_KEY)
	{
		results = originalResults;
	}

	HK_TIMER_END();
	return bestKey != HK_INVALID_SHAPE_KEY;
}

void hkpListShape::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
// copy of castRayImpl() with modifications

	HK_TIMER_BEGIN("rcList",HK_NULL);

	if ( !input.m_rayShapeCollectionFilter )
	{
		for (int i = 0; i < m_childInfo.getSize(); i++)
		{
			if (isChildEnabled(i))
			{
#if !defined(HK_PLATFORM_SPU)
				const hkpShape* childShape = m_childInfo[i].m_shape;					
#else
				hkpShapeBuffer shapeBuffer;
				const hkpShape* childShape = getChildShape(i, shapeBuffer);
#endif
				hkpCdBody childBody(&cdBody);
				childBody.setShape(childShape, i);
				childShape->castRayWithCollector( input, childBody, collector );
			}
		}
	}
	else
	{
		for (int i = 0; i < m_childInfo.getSize(); i++)
		{
#if !defined(HK_PLATFORM_SPU)
			if ( isChildEnabled(i) && (false!=input.m_rayShapeCollectionFilter->isCollisionEnabled( input, *this, i )) )
#else
			if ( isChildEnabled(i) && hkpSpuCollisionFilterUtil::s_rayShapeContainerIsCollisionEnabled( input, *this, i) )
#endif
			{
#if !defined(HK_PLATFORM_SPU)
				const hkpShape* childShape = m_childInfo[i].m_shape;					
#else
				hkpShapeBuffer shapeBuffer;
				const hkpShape* childShape = getChildShape(i, shapeBuffer);
#endif
				hkpCdBody childBody(&cdBody);
				childBody.setShape(childShape, i);
				childShape->castRayWithCollector( input, childBody, collector );
			}
		}
	}

	HK_TIMER_END();
}

void hkpListShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out ) const
{
#if !defined(HK_PLATFORM_SPU)

	// On cpu/ppu we will recalculate the hkpListShape's AABB on the fly by calculating the AABBs of its
	// children. The reason for this is that the costs for this function are much smaller than dealing
	// with an too large AABB.

	// Set up for SPU. <<sk.todo.aa allow serialization to do this calc on load.
	if ( m_aabbHalfExtents.lessEqualZero().allAreSet() )
	{
		(const_cast<hkpListShape*>(this))->recalcAabbExtents();
	}

	out.m_min.setAll( hkSimdReal_Max*hkSimdReal_Half ); 
	out.m_max.setNeg<4>( out.m_min);

	hkAabb t;
	for (int i = 0; i < m_childInfo.getSize(); i++)
	{
		const hkpShape* childShape = m_childInfo[i].m_shape;
		childShape->getAabb( localToWorld, tolerance, t );
		out.m_min.setMin( out.m_min, t.m_min );
		out.m_max.setMax( out.m_max, t.m_max );
	}

#else

	// On spu we will not recalculate the children's AABBs separately but only use the initial (cached) overall AABB.
	// This will give us slightly worse results but saves us from bringing in the children onto the spu for this calculation.
	hkAabbUtil::calcAabb( localToWorld, m_aabbHalfExtents, m_aabbCenter, hkSimdReal::fromFloat(tolerance), out );

#endif
}


#if defined (HK_PLATFORM_SPU)

void hkpListShape::getAabbWithChildShapes(const hkpCollisionInput& collisionInput, const hkAabbUtil::OffsetAabbInput& input, const hkTransform& localToWorld, const hkVector4& massCenter, hkReal tolerance, hkAabbUint32& rootAabbUint32, hkAabbUint32* childShapesAabbUint32, hkpShapeKey* childShapeKeys, int numAabbs, int sizeOfAabbBuffer_usedOnSpu) const
{
	HK_ON_DEBUG(int enabledChildCount = 0);
	HK_ON_DEBUG(int numChildShapes = hkpListShape::getNumChildShapes());
	const int numChildInfos = m_childInfo.getSize(); 
	HK_ASSERT(0xaf35efe2, numChildShapes <= numAabbs);
	HK_ASSERT(0xaf35efe3, numChildShapes < 256 );

	//
	// Allocate memory for up to numChildShapes. We will also use this AABB memory as
	// a buffer for the ChildInfo data. For this we will place the ChildInfo data at the end of the buffer while we
	// will begin filling the buffer with AABBs from the start.
	//
	ChildInfo* childInfoBuffer;
	{
		int sizeOfChildInfoBuffer = numChildInfos * sizeof(ChildInfo);
		hkAabbUint32* childShapesAabbUint32BufferEnd = hkAddByteOffset(childShapesAabbUint32, sizeOfAabbBuffer_usedOnSpu);
		childInfoBuffer = reinterpret_cast<ChildInfo*>( hkUlong(childShapesAabbUint32BufferEnd) - hkUlong(sizeOfChildInfoBuffer) );
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(childInfoBuffer, m_childInfo.begin(), sizeOfChildInfoBuffer, hkSpuDmaManager::READ_COPY);
		HK_SPU_DMA_PERFORM_FINAL_CHECKS(m_childInfo.begin(), childInfoBuffer, sizeOfChildInfoBuffer);
	}

	//
	// Allocate and initialize the pipeline for bringing in the child shapes.
	//
	hkListShapeGetAabbWithChildShapes::Pipeline* pipeline = hkAllocateStack<hkListShapeGetAabbWithChildShapes::Pipeline>(1, "hkpListShape::gAWCS::Pipe"); // See 0xaf5241e7.
	pipeline->init(0); // we can use whatever DMA group we want as we have just completely stalled everything

	hkListShapeGetAabbWithChildShapes::PipelineStage* getShapesStage	= &pipeline->m_stages[0];
	hkListShapeGetAabbWithChildShapes::PipelineStage* calcAabbStage	= &pipeline->m_stages[1];

	int numBlocksTotal = ((numChildInfos-1) / SHAPE_BLOCK_SIZE) + 1;
	int numShapesLeft = numChildInfos;
		
	hkAabb rootAabb; rootAabb.setEmpty();
	hkAabb rootSweptAabb = rootAabb;

	int shapeKey = 0;
	{
		// loop 1x more to make sure that the last shape block actually finishes through the last pipeline stage
		for (int blockIdx = 0; blockIdx < numBlocksTotal+1; blockIdx++)
		{
			//
			// STAGE 0 : get shape block
			//
			if ( hkMath::intInRange( blockIdx, 0, numBlocksTotal+0 ) )
			{
				// stage-local data
				hkListShapeGetAabbWithChildShapes::PipelineStage*	stageData		= getShapesStage;
				int													stageDmaGroup	= stageData->m_dmaGroup;

				int numShapesLeftForBlock = hkMath::min2(numShapesLeft, SHAPE_BLOCK_SIZE);
				stageData->m_numShapesInBlock = numShapesLeftForBlock;

				{
					int shapeIdxInBlock = 0;
					do
					{
						HK_ASSERT2(0xad808151, hkUlong(childShapesAabbUint32) < hkUlong(childInfoBuffer), "Data corruption.");

						// Only handle enabled child shapes.
						if ( isChildEnabled(shapeKey) )
						{
							// Get the shape from PPU.
							hkSpuDmaManager::getFromMainMemory(stageData->getShape(shapeIdxInBlock), childInfoBuffer->m_shape, HK_SPU_MAXIMUM_SHAPE_SIZE, hkSpuDmaManager::READ_COPY, stageDmaGroup);
							HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT(childInfoBuffer->m_shape, stageData->getShape(shapeIdxInBlock), HK_SPU_MAXIMUM_SHAPE_SIZE);

							// Store information that is needed by a later stage.
							stageData->m_aabbs[shapeIdxInBlock]     = childShapesAabbUint32;
							stageData->m_aabbs[shapeIdxInBlock]->m_shapeKeyByte	= hkUint8(shapeKey);
							*childShapeKeys = shapeKey;

							// Advance to next child shape info and destination AABB.
							childShapesAabbUint32	= hkAddByteOffset(childShapesAabbUint32, sizeof(hkAabbUint32));
							childShapeKeys			= hkAddByteOffset(childShapeKeys, sizeof(hkpShapeKey));
							shapeIdxInBlock++;
							HK_ON_DEBUG(enabledChildCount++);

						}
						else
						{
							//numShapesLeftForBlock = hkMath::min2(numShapesLeft, numShapesLeftForBlock);
							HK_ASSERT2(0xad808209, numShapesLeftForBlock <= numShapesLeft, "Internal error.");
							numShapesLeftForBlock--;
						}

						numShapesLeft--;
						childInfoBuffer++;
						shapeKey++;

					} while( shapeIdxInBlock < numShapesLeftForBlock );

					stageData->m_numShapesInBlock = shapeIdxInBlock;
				}
			}

			//
			// STAGE 1 : calc AABBs
			// (only perform this stage starting with the 2nd iteration and make sure that the last entity will still pass this stage)
			//
			if ( hkMath::intInRange( blockIdx, 1, numBlocksTotal+1 ) )
			{
				// stage-local data
				hkListShapeGetAabbWithChildShapes::PipelineStage*	stageData		= calcAabbStage;
				int													stageDmaGroup	= stageData->m_dmaGroup;

				// wait for SHAPES to arrive
				hkSpuDmaManager::waitForDmaCompletion( stageDmaGroup );

				int numShapesInBlock = stageData->m_numShapesInBlock;
				{
					int shapeIdxInBlock = 0;
					while( shapeIdxInBlock < numShapesInBlock )
					{
						hkAabbUint32* aabb32   = stageData->m_aabbs[shapeIdxInBlock];
						hkUint8 shapeKeyByte = aabb32->m_shapeKeyByte;

						hkAabb aabb;
						hkpShape* shape = stageData->getShape(shapeIdxInBlock);
						HKCD_PATCH_SHAPE_VTABLE( shape );
						shape->getAabb(localToWorld, tolerance, aabb);

						hkAabb sweptAabb;
						hkAabbUtil::sweepOffsetAabb(input, aabb, sweptAabb);

						// Add to the root-collidable AABBs
						rootAabb.m_min.setMin(rootAabb.m_min, aabb.m_min);
						rootAabb.m_max.setMax(rootAabb.m_max, aabb.m_max);
						rootSweptAabb.m_min.setMin(rootSweptAabb.m_min, sweptAabb.m_min);
						rootSweptAabb.m_max.setMax(rootSweptAabb.m_max, sweptAabb.m_max);

						// Convert to integer space and compress
						hkAabbUtil::convertAabbToUint32(aabb,      collisionInput.m_aabb32Info.m_bitOffsetLow, collisionInput.m_aabb32Info.m_bitOffsetHigh, collisionInput.m_aabb32Info.m_bitScale, *aabb32);

						hkAabbUint32 sweptAabbUint32;
						hkAabbUtil::convertAabbToUint32(sweptAabb, collisionInput.m_aabb32Info.m_bitOffsetLow, collisionInput.m_aabb32Info.m_bitOffsetHigh, collisionInput.m_aabb32Info.m_bitScale, sweptAabbUint32 );
						hkAabbUtil::compressExpandedAabbUint32(sweptAabbUint32, *aabb32);

						// Write the shapeKey back.
						aabb32->m_shapeKeyByte = shapeKeyByte;

						shapeIdxInBlock++;

					}
				}
			}

			//
			// rotate the 2 "stage" buffers
			//
			{
				hkListShapeGetAabbWithChildShapes::PipelineStage* buffer	= calcAabbStage;
				calcAabbStage												= getShapesStage;
				getShapesStage												= buffer;
			}

		}
	}

	// If all children are disabled the center of mass is used to obtain a valid root AABB
	hkVector4Comparison childrenEnabled = rootAabb.m_max.greaterEqual(rootAabb.m_min);
	rootAabb.m_max.setSelect(childrenEnabled, rootAabb.m_max, massCenter);
	rootAabb.m_min.setSelect(childrenEnabled, rootAabb.m_min, massCenter);
	rootSweptAabb.m_max.setSelect(childrenEnabled, rootSweptAabb.m_max, massCenter);
	rootSweptAabb.m_min.setSelect(childrenEnabled, rootSweptAabb.m_min, massCenter);

	// Convert root-collidable AABB to integer space too.
	hkAabbUint32 rootSweptAabbUint32;
	hkAabbUtil::convertAabbToUint32(rootAabb,      collisionInput.m_aabb32Info.m_bitOffsetLow, collisionInput.m_aabb32Info.m_bitOffsetHigh, collisionInput.m_aabb32Info.m_bitScale, rootAabbUint32     );
	hkAabbUtil::convertAabbToUint32(rootSweptAabb, collisionInput.m_aabb32Info.m_bitOffsetLow, collisionInput.m_aabb32Info.m_bitOffsetHigh, collisionInput.m_aabb32Info.m_bitScale, rootSweptAabbUint32);
	hkAabbUtil::compressExpandedAabbUint32(rootSweptAabbUint32, rootAabbUint32);

	hkDeallocateStack(pipeline);

	HK_ASSERT2(0xad808145, enabledChildCount == numAabbs, "Num requested aabbs and actual number of enabled children doesn't match.");
}

#endif // defined(HK_PLATFORM_SPU)


HK_COMPILE_TIME_ASSERT( sizeof(hkAabbUint32) == sizeof(hk1AxisSweep::AabbInt) );

#if defined (HK_PLATFORM_SPU)

int hkpListShape::getAabbWithChildShapesForAgent(const hkpCollisionInput& collisionInput, const hkAabbUtil::OffsetAabbInput& input, hkBool32 useContinuousPhysics, const hkTransform& localToWorld, hkReal tolerance, hkAabb& rootAabb, hk1AxisSweep::AabbInt* aabbs, int numAabbs, int sizeOfAabbBuffer_usedOnSpu) const
{
	const int numChildInfos = m_childInfo.getSize();
	HK_ASSERT(0xaf35efe2, hkpListShape::getNumChildShapes() <= numAabbs);
	HK_ASSERT(0xaf35efe3, m_childInfo.getSize() < 256 );

	//
	// Allocate memory for up to numChildShapes. We will also use this AABB memory as
	// a buffer for the ChildInfo data. For this we will place the ChildInfo data at the end of the buffer while we
	// will begin filling the buffer with AABBs from the start.
	//
	ChildInfo* childInfoBuffer;
	{
		const int sizeOfChildInfoBuffer = numChildInfos * sizeof(ChildInfo);
		hkAabbUint32* childShapesAabbUint32BufferEnd = hkAddByteOffset(aabbs, sizeOfAabbBuffer_usedOnSpu);
		childInfoBuffer = reinterpret_cast<ChildInfo*>( hkUlong(childShapesAabbUint32BufferEnd) - hkUlong(sizeOfChildInfoBuffer) );
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(childInfoBuffer, m_childInfo.begin(), sizeOfChildInfoBuffer, hkSpuDmaManager::READ_COPY);
		HK_SPU_DMA_PERFORM_FINAL_CHECKS(m_childInfo.begin(), childInfoBuffer, sizeOfChildInfoBuffer);
	}

	//
	// Allocate and initialize the pipeline for bringing in the child shapes.
	//
	hkListShapeGetAabbWithChildShapesForAgent::Pipeline* pipeline = hkAllocateStack<hkListShapeGetAabbWithChildShapesForAgent::Pipeline>(1, "hkpListShape::gAWCSFAgent::Pipe"); // See 0xaf5241e8.
	pipeline->init(0); // we can use whatever DMA group we want as we have just completely stalled everything

	hkListShapeGetAabbWithChildShapesForAgent::PipelineStage* getShapesStage	= &pipeline->m_stages[0];
	hkListShapeGetAabbWithChildShapesForAgent::PipelineStage* calcAabbStage		= &pipeline->m_stages[1];

	hkAabb aabb;
	rootAabb.setEmpty(); //we're not setting this to centerOfMassInWorld because this is a 'for agent' function, and this rootAabb won't be used at all.

	int enabledShapesCtr = 0;

	{
		int numShapesDisabled = 0;
		// Loop 1x more to make sure that the last shape actually finishes through the last pipeline stage.
		for (int shapeIdx = 0; shapeIdx < numChildInfos+1; shapeIdx++)
		{
			//
			// STAGE 0 : Get shape.
			//
			if ( hkMath::intInRange( shapeIdx, 0, numChildInfos+0 ) )
			{
				// stage-local data
				hkListShapeGetAabbWithChildShapesForAgent::PipelineStage*	stageData		= getShapesStage;
				int															stageDmaGroup	= stageData->m_dmaGroup;

				// Only handle enabled child shapes.
				// As we have are a hkpListShape, shapeIdx can be used as shapeKey.
				if ( hkpListShape::isChildEnabled(shapeIdx) )
				{
					// Get the shape from PPU.
					hkSpuDmaManager::getFromMainMemory(stageData->getShape(), childInfoBuffer->m_shape, HK_SPU_MAXIMUM_SHAPE_SIZE, hkSpuDmaManager::READ_COPY, stageDmaGroup);
					HK_SPU_DMA_DEFER_FINAL_CHECKS_UNTIL_WAIT(childInfoBuffer->m_shape, stageData->getShape(), HK_SPU_MAXIMUM_SHAPE_SIZE);

					// Store information that is needed by a later stage.
					stageData->m_aabb         = aabbs;
					stageData->m_shapeKey     = shapeIdx;

					// Advance to next free AABB in array.
					aabbs++;

					// Advance to next child shape info.
					childInfoBuffer++;
				}
				else
				{

					// Advance to next child shape info.
					childInfoBuffer++;
					numShapesDisabled++;
					continue;
				}

			}

			//
			// STAGE 1 : Calculate AABB.
			// (only perform this stage starting with the 2nd iteration and make sure that the last shape will still pass this stage)
			//
			if ( hkMath::intInRange( shapeIdx, 1+numShapesDisabled, numChildInfos+1 ) )
			{
				// stage-local data
				hkListShapeGetAabbWithChildShapesForAgent::PipelineStage*	stageData		= calcAabbStage;
				int															stageDmaGroup	= stageData->m_dmaGroup;

				// Wait for SHAPE to arrive.
				hkSpuDmaManager::waitForDmaCompletion( stageDmaGroup );

				hkpShape* shape = stageData->getShape();
				HKCD_PATCH_SHAPE_VTABLE( shape );
				shape->getAabb(localToWorld, tolerance, aabb);

				if ( useContinuousPhysics )
				{
					hkAabbUtil::sweepOffsetAabb(input, aabb, aabb);
				}

				// Add to the root-collidable AABBs
				rootAabb.m_min.setMin(rootAabb.m_min, aabb.m_min);
				rootAabb.m_max.setMax(rootAabb.m_max, aabb.m_max);

				// Convert to integer space.
				hkAabbUint32* aabb32 = stageData->m_aabb;
				hkAabbUtil::convertAabbToUint32(aabb, collisionInput.m_aabb32Info.m_bitOffsetLow, collisionInput.m_aabb32Info.m_bitOffsetHigh, collisionInput.m_aabb32Info.m_bitScale, *aabb32);

				(static_cast<hk1AxisSweep::AabbInt*>(aabb32))->getKey() = stageData->m_shapeKey;

				enabledShapesCtr++;
			}

			//
			// rotate the 2 "stage" buffers
			//
			{
				hkListShapeGetAabbWithChildShapesForAgent::PipelineStage* buffer	= calcAabbStage;
				calcAabbStage														= getShapesStage;
				getShapesStage														= buffer;
			}

		}
	}

	hkDeallocateStack(pipeline);

	HK_ASSERT2(0xad808152, enabledShapesCtr <= numAabbs, "Actual num of aabbs read must be less or equal numAabbs.");

	return enabledShapesCtr;
}

int hkpListShape::getAabbWithChildShapesForAgent_withNoDmas(const hkpCollisionInput& collisionInput, const hkAabbUtil::OffsetAabbInput& input, hkBool32 useContinuousPhysics, const hkTransform& localToWorld, hkReal tolerance, hkAabb& rootAabb, hk1AxisSweep::AabbInt* aabbs, int numAabbs) const
{
	HK_ASSERT(0xaf35efe2, m_childInfo.getSize() == numAabbs && numAabbs == 1);
	HK_ASSERT(0xaf35efe3, isChildEnabled(0));

	const hkpShape* oneChild = m_childInfo[0].m_shape;

	oneChild->getAabb(localToWorld, tolerance, rootAabb);

	if ( useContinuousPhysics )
	{
		hkAabbUtil::sweepOffsetAabb(input, rootAabb, rootAabb);
	}

	// Convert to integer space.
	hkAabbUint32* aabb32 = aabbs;
	hkAabbUtil::convertAabbToUint32(rootAabb, collisionInput.m_aabb32Info.m_bitOffsetLow, collisionInput.m_aabb32Info.m_bitOffsetHigh, collisionInput.m_aabb32Info.m_bitScale, *aabb32);

	(static_cast<hk1AxisSweep::AabbInt*>(aabb32))->getKey() = 0;

	return 1;
}

#endif


void hkpListShape::recalcAabbExtents( )
{
	hkAabb aabb;
	recalcAabbExtents( aabb );
}

void hkpListShape::recalcAabbExtents( hkAabb& aabb )
{
	m_childInfo[0].m_shape->getAabb( hkTransform::getIdentity(), 0.0f, aabb );

	hkAabb t;
	{
		for (int i = 1; i < m_childInfo.getSize(); i++)
		{
			m_childInfo[i].m_shape->getAabb( hkTransform::getIdentity(), 0.0f, t );
			aabb.m_min.setMin( aabb.m_min, t.m_min );
			aabb.m_max.setMax( aabb.m_max, t.m_max );
		}
	}
	aabb.getCenter( m_aabbCenter );
	aabb.getHalfExtents( m_aabbHalfExtents );
}


#if defined(HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Spu/Dma/Utils/hkSpuDmaUtils.h>
#endif

const hkpShape* hkpListShape::getChildShape(hkpShapeKey key, hkpShapeBuffer& buffer) const
{
#if !defined(HK_PLATFORM_SPU)
	return m_childInfo[ key ].m_shape;
#else

	const void* shapeOnPpu;
	int shapeOnPpuSize;
	if ( m_flags & DISABLE_SPU_CACHE_FOR_LIST_CHILD_INFO )
	{
		// HVK-3938 means we can't use the cache here. Timings show the cache makes almost no difference here
		// First DMA over the childInfo so that we can access the shape ptrs and the size
		ChildInfo childInfo;
		const void* childInfoOnPpu = hkAddByteOffset( (void*)m_childInfo.begin() ,  sizeof(ChildInfo) * key );
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( &childInfo, childInfoOnPpu, sizeof(ChildInfo), hkSpuDmaManager::READ_COPY );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS( childInfoOnPpu, &childInfo, sizeof(ChildInfo) );
		shapeOnPpu = childInfo.m_shape;
		shapeOnPpuSize = hkUint16(childInfo.m_shapeSize);
	}
	else
	{
		HK_CRITICAL_ASSERT(0x5171afef, (int) key < m_childInfo.getSize() );
		const ChildInfo* childInfo = hkGetArrayElemUsingCache( m_childInfo.begin(), key, g_SpuCollideUntypedCache, HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE );
		shapeOnPpu = childInfo->m_shape;
		shapeOnPpuSize = hkUint16(childInfo->m_shapeSize);
	}

	// Finally DMA over the shape
	const hkpShape* shape;
	{
		const void* shapeOnSpu = g_SpuCollideUntypedCache->getFromMainMemory(shapeOnPpu, shapeOnPpuSize);
		// COPY over to buffer (instead of dmaing to buffer above, since we are returning this data)
		hkString::memCpy16NonEmpty( buffer, shapeOnSpu, ((shapeOnPpuSize+15)>>4) );
		shape = reinterpret_cast<hkpShape*>(buffer);
	}

	HKP_PATCH_CONST_SHAPE_VTABLE( shape );
	return shape;

#endif
}

hkUint32 hkpListShape::getCollisionFilterInfo(hkpShapeKey key) const
{
#if !defined(HK_PLATFORM_SPU)
	return m_childInfo[ key ].m_collisionFilterInfo;
#else
	if ( m_flags & DISABLE_SPU_CACHE_FOR_LIST_CHILD_INFO )
	{
		// HVK-3938 means we can't use the cache here. Timings show the cache makes almost no difference here
		// First DMA over the childInfo so that we can access the shape ptrs and the size
		ChildInfo childInfo;
		const void* childInfoOnPpu = hkAddByteOffset( (void*)m_childInfo.begin() ,  sizeof(ChildInfo) * key );
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( &childInfo, childInfoOnPpu, sizeof(ChildInfo), hkSpuDmaManager::READ_COPY );
		return childInfo.m_collisionFilterInfo;
	}
	else
	{
		const ChildInfo* childInfo = hkGetArrayElemUsingCache( m_childInfo.begin(), key, g_SpuCollideUntypedCache, HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE );
		return childInfo->m_collisionFilterInfo;
	}
#endif
}

#if !defined(HK_PLATFORM_SPU)

void hkpListShape::setCollisionFilterInfo( hkpShapeKey index, hkUint32 filterInfo )
{
	m_childInfo[ index ].m_collisionFilterInfo = filterInfo;
}

int hkpListShape::calcSizeForSpu(const CalcSizeForSpuInput& input, int spuBufferSizeLeft) const
{
	if ( input.m_midphaseAgent3Registered && !input.m_isFixedOrKeyframed && m_childInfo.getSize() > MAX_CHILDREN_FOR_SPU_MIDPHASE )
	{
		HK_WARN(0xf0345d7, "This ListShape has too many children to be processed on the SPU using the collection collection agent");
		return -1;
	}

	if ( m_childInfo.getSize() >= HK_MAX_NUM_HITS_PER_AABB_QUERY-1 )
	{
		HK_WARN(0xf0345d1, "This ListShape has too many children to be processed on the SPU");
		return -1;
	}

	for (int i = 0; i < m_childInfo.getSize(); i++)
	{
		m_childInfo[i].m_shapeSize = (hkInt16)m_childInfo[i].m_shape->calcSizeForSpu(input, HK_SPU_AGENT_SECTOR_JOB_MAX_SHAPE_SIZE);
		m_childInfo[i].m_numChildShapes = m_childInfo.getSize();
		if ( m_childInfo[i].m_shapeSize == -1 )
		{
			// early out if one of our children is not supported on spu
			HK_WARN(0xdbc05911, "Listshape child " << i << " cannot be processed on SPU.");
			return -1;
		}
	}

	return sizeof(*this);
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

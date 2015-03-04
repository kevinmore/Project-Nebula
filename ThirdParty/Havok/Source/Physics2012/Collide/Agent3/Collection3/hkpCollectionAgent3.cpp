/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#if defined(HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Monitor/Spu/hkSpuMonitorCache.h>
#endif
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Filter/hkpShapeCollectionFilter.h>
#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Physics2012/Collide/Agent3/BvTree3/hkpBvTreeAgent3.h>
#include <Physics2012/Collide/Agent3/Collection3/hkpCollectionAgent3.h>
#include <Physics2012/Collide/Agent3/CollectionCollection3/hkpCollectionCollectionAgent3.h>
#include <Physics2012/Collide/Agent3/List3/hkpListAgent3.h>

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>
#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpShapeKeyTrack.h>

void hkpCollectionAgent3::initAgentFunc(hkpCollisionDispatcher::Agent3Funcs& f)
{
	hkListAgent3::initAgentFunc(f);

	f.m_processFunc  = process;
}

#if !defined(HK_PLATFORM_SPU)
void hkpCollectionAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	initAgentFunc(f);
	dispatcher->registerAgent3( f, hkcdShapeType::CONVEX, hkcdShapeType::COLLECTION );
}
#endif

HK_FORCE_INLINE void hkpCollectionAgent3process_before( const hkpAgent3ProcessInput& input, hkUint8* shapeBuffer, hkUint8* collidableBuffer, hkpAgent3ProcessInput& modifiedInput )
{
	const hkpShape* thisSingleShape = input.m_bodyA->m_shape;				// Get the non-list shape

#	if ! defined (HK_PLATFORM_SPU)
	hkpCollidable* collidable = new (collidableBuffer) hkpCollidable(HK_NULL, (hkMotionState*)HK_NULL); // needed to access bvInformation
	hkpListShape* list = new (shapeBuffer) hkpListShape(&thisSingleShape, 1, hkpShapeContainer::REFERENCE_POLICY_IGNORE);
	if (!input.m_bodyA->m_parent)
	{
		const hkpCollidable* originalCollidable = static_cast<const hkpCollidable*>(input.m_bodyA.val());
		collidable->m_boundingVolumeData = originalCollidable->m_boundingVolumeData;
	}
	hkpCdBody* modifiedCdBodyA = new (static_cast<hkpCdBody*>(collidable)) hkpCdBody(input.m_bodyA->m_parent, input.m_bodyA->getMotionState());
#	else
	const int sizeof_hkpListShape = HK_NEXT_MULTIPLE_OF(16, sizeof(hkpListShape));
	hkString::memSet16(shapeBuffer, &hkVector4::getZero(), sizeof_hkpListShape >> 4);
	hkpListShape* list = reinterpret_cast<hkpListShape*>(shapeBuffer);
	list->m_type = hkcdShapeType::LIST;
	list->m_enabledChildren[0] = unsigned(-1);
	HK_ALIGN16(hkpListShape::ChildInfo arrayElement);
	HK_ASSERT(0XAD873433, sizeof(hkpListShape::ChildInfo) == 16);
	hkString::memSet16(&arrayElement, &hkVector4::getZero(), sizeof(hkpListShape::ChildInfo) >> 4);
	new (&list->m_childInfo) hkArray<hkpListShape::ChildInfo>(&arrayElement, 1, 1);
	list->m_childInfo[0].m_shape = thisSingleShape;
	list->m_childInfo[0].m_numChildShapes = 1;

	hkpCollidable* collidablePtr = reinterpret_cast<hkpCollidable*>(collidableBuffer);
	collidablePtr->m_owner = HK_NULL;
	if (!input.m_bodyA->m_parent)
	{
		const hkpCollidable* originalCollidable = static_cast<const hkpCollidable*>(input.m_bodyA.val());
		HK_ASSERT2(0xad873643, (((hkUlong)(&collidablePtr->m_boundingVolumeData) & 0x0ful) == 0) && (((hkUlong)(&originalCollidable->m_boundingVolumeData) & 0x0ful) == 0), "Bounding volume data is expected to be 16-byte aligned.");
		const int sizeOfBoundingVolumeData = HK_NEXT_MULTIPLE_OF(16, sizeof(hkCollidablePpu::BoundingVolumeData));
		hkString::memCpy16(&collidablePtr->m_boundingVolumeData, &originalCollidable->m_boundingVolumeData, sizeOfBoundingVolumeData >> 4);
	}
	hkpCdBody* modifiedCdBodyA = new (static_cast<hkpCdBody*>(collidablePtr)) hkpCdBody(input.m_bodyA->m_parent, input.m_bodyA->getMotionState());
#	endif

	modifiedCdBodyA->setShape(list, HK_INVALID_SHAPE_KEY);		

	HK_ASSERT2(0xad893433, !modifiedInput.m_overrideBodyA, "This may be an incorrect assert. Just testing.");
	modifiedInput.m_overrideBodyA = input.m_bodyA;
	modifiedInput.m_bodyA = modifiedCdBodyA;
}

HK_FORCE_INLINE void hkpCollectionAgent3process_after( hkUint8* shapeBuffer )
{
#if ! defined( HK_PLATFORM_SPU )
	hkpListShape* list = reinterpret_cast<hkpListShape*>( shapeBuffer );
	list->m_childInfo.setSize(0);
	list->~hkpListShape();
#endif
}

hkpAgentData* hkpCollectionAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	HK_TIMER_BEGIN( "Coll3", HK_NULL );

	HK_ALIGN_REAL( hkUint8 shapeBuffer[HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hkpListShape))] );
	HK_ALIGN_REAL( hkUint8 collidableBuffer[HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, (sizeof(hkpCollidable)))] );
	hkpAgent3ProcessInput modifiedInput = input;
	
	hkpCollectionAgent3process_before( input, shapeBuffer, collidableBuffer, modifiedInput );

	hkpAgentData* result = hkpCollectionCollectionAgent3::process(modifiedInput, entry, agentData, separatingNormal, output);

	hkpCollectionAgent3process_after( shapeBuffer );

	HK_TIMER_END();

	return result;
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

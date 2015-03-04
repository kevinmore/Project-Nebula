/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/hkpShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>

#if defined(HK_PLATFORM_HAS_SPU)
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#endif

hkBool hkpCollidable::checkPerformance()
{
	if(!m_shape)
	{
		return false;
	}

	// THE CHECKS BELOW ARE DUPLICATED IN hkpRigidBody::checkPerformance() TO BE SUPER SAFE. IF YOU CHANGE/ADD STUFF HERE
	// PLEASE SEE IF YOU NEED TO ALSO DO THE SAME IN hkpRigidBody::checkPerformance()

	hkBool isOk = true;
	const int manyChildren = 100;

	// List (with many children)
	if( m_shape->getType() == hkcdShapeType::LIST 
		&& static_cast<const hkpListShape*>(m_shape)->getNumChildShapes() > manyChildren ) 
	{
		HK_WARN(0x2ff8c16c, "Collidable at address " << this << " has an hkpListShape with > " << manyChildren << " children.\n" \
			"This can cause a significant performance loss when the shape is collided e.g. with another complex hkpListShape.\n" \
			"When using hkpListShape with many children consider using building an hkpMoppBvTreeShape around it.\n");
		isOk = false;
	}
	
	// Mesh (without MOPP)
	if( m_shape->getType() == hkcdShapeType::TRIANGLE_COLLECTION && static_cast<const hkpShapeCollection*>(m_shape)->getNumChildShapes() > manyChildren )
	{
		HK_WARN(0x2ff8c16e, "Collidable at address " << this << " has a mesh shape.\n" \
			"This can cause a significant performance loss.\n" \
			"The collection of triangle shapes (hkpMeshShape) should not be used for dynamic bodies.\n" \
			"You should consider building an hkpMoppBvTreeShape around the mesh.\n");
		isOk = false;
	}

	// Collection (without MOPP)
	if( (m_shape->getType() == hkcdShapeType::COLLECTION || m_shape->getType() == hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION)
		&& static_cast<const hkpShapeCollection*>(m_shape)->getNumChildShapes() > manyChildren )
	{
		HK_WARN(0x578cef50, "Collidable at address " << this << " has a shape collection without a hkpBvTreeShape.\n" \
			"This can cause performance loss. To avoid getting this message\n" \
			"add a hkpBvTreeShape above this shape.");
		isOk = false;
	}

	// Transformed shape
	if( m_shape->getType() == hkcdShapeType::TRANSFORM)
	{
		HK_WARN(0x2ff8c16f, "Collidable at address " << this << " has a transform shape as the root shape.\n" \
				"This can cause a significant performance loss. To avoid getting this message\n" \
				"compose the transform into the collidable and remove the transform shape.\n" \
				"Please see the 'hkpTransformShape' documentation in the User Guide for more information.\n");
		isOk = false;
	}
	
	return isOk;
}


hkpCollidable::BoundingVolumeData::BoundingVolumeData()
{
	hkString::memSet( this, 0, sizeof(*this) );
	invalidate();
}


void hkpCollidable::BoundingVolumeData::deallocate()
{
	HK_ASSERT2(0xaf35fe31, hasAllocations(), "Deallocating non-existent structure.");
	hkDeallocateChunk(m_childShapeAabbs, m_capacityChildShapeAabbs, HK_MEMORY_CLASS_COLLIDE);
	hkDeallocateChunk(m_childShapeKeys, HK_NEXT_MULTIPLE_OF(16/sizeof(hkpShapeKey),m_capacityChildShapeAabbs), HK_MEMORY_CLASS_COLLIDE);
	m_childShapeAabbs    = HK_NULL;
	m_childShapeKeys    = HK_NULL;
	m_numChildShapeAabbs = 0;
	m_capacityChildShapeAabbs = 0;
}

void hkpCollidable::BoundingVolumeData::allocate(int numChildShapes)
{
	HK_ASSERT2(0xad808123, !hasAllocations(), "Structure already has allocation.");
	HK_ASSERT2(0xad808124, numChildShapes <= hkUint16(-1), "Too many child shapes (maximum = 2^16). Did you forget to wrap a shape collection with a MOPP?" );
	m_numChildShapeAabbs = hkUint16(numChildShapes);
	m_capacityChildShapeAabbs = hkUint16(numChildShapes);
	if ( numChildShapes )
	{
		m_childShapeAabbs    = hkAllocateChunk<hkAabbUint32>(m_capacityChildShapeAabbs, HK_MEMORY_CLASS_COLLIDE);
		m_childShapeKeys     = hkAllocateChunk<hkpShapeKey>( HK_NEXT_MULTIPLE_OF(16/sizeof(hkpShapeKey),m_capacityChildShapeAabbs), HK_MEMORY_CLASS_COLLIDE);
	}
}

#if defined (HK_PLATFORM_HAS_SPU)
void hkpCollidable::setShapeSizeForSpu()
{
	const hkpShape* shape = getShape();
	if ( shape )
	{
		m_forceCollideOntoPpu &= ~hkpCollidable::FORCE_PPU_SHAPE_UNCHECKED;

		hkpShape::CalcSizeForSpuInput input;
		input.m_midphaseAgent3Registered = false;  // Not relevant for phantoms, so set to false. 
		input.m_isFixedOrKeyframed = false; // Not relevant for phantoms, so set to false.

		// Change input if the owner of collidable is a rigid body;
		hkpRigidBody* rigidBody = hkpGetRigidBody(this);
		if (rigidBody)
		{
			// if rigid body
			input.m_isFixedOrKeyframed = rigidBody->isFixedOrKeyframed();
			input.m_hasDynamicMotionSaved = (HK_NULL != rigidBody->getMotion()->m_savedMotion);		

			// If the rigid body has not been added to the world, and world is HK_NULL, then assume
			// that the CollectionCollection3 agent is registered.
			hkpWorld* world = rigidBody->getWorld();
			if( world == HK_NULL )
			{
				input.m_midphaseAgent3Registered = true;
				HK_WARN_ONCE(0x56f80833, "Cannot check if midphase agent is registered because entity is not in the world.  Assuming that midphase agent IS registered when calculating shape size for SPU." );
				HK_WARN_ONCE(0x56f80834, "Cannot check if compressed shape is registered to run on SPU because entity is not in the world. Assuming it ISN'T registered when calculating shape size for SPU." );
			}
			else
			{
				input.m_midphaseAgent3Registered = world->getCollisionDispatcher()->m_midphaseAgent3Registered;
			}
		}

		int shapeSize = shape->calcSizeForSpu(input, HK_SPU_AGENT_SECTOR_JOB_MAX_SHAPE_SIZE);
		if ( shape->m_type == hkcdShapeType::EXTENDED_MESH || shape->m_type == hkcdShapeType::COMPRESSED_MESH  || shape->m_type == hkcdShapeType::TRIANGLE_COLLECTION )
		{
			HK_WARN_ONCE(0x5607bb49, "Mesh shape used without an hkpBvTreeShape. Forcing on to PPU. Possible huge performance loss");
			shapeSize = -1;
		}

		// we have to force the object's collision detection completely onto PPU if:
		// (a) calcSizeForSpu() returns -1 (if the shape or one of its potential children wasn't updated to work
		//     with this function yet or - in case of shape cascades - if the shape size is too large to fit into
		//     the spu's shape buffer)
		// (b) the shape's total size (incl. potential children) is too large to fit into the spu's shape buffer
		if ( shapeSize < 0 || shapeSize > HK_SPU_AGENT_SECTOR_JOB_MAX_SHAPE_SIZE )
		{
			m_forceCollideOntoPpu |= hkpCollidable::FORCE_PPU_SHAPE_REQUEST;
		}
		else
		{
			m_forceCollideOntoPpu &= ~hkpCollidable::FORCE_PPU_SHAPE_REQUEST;
			m_shapeSizeOnSpu = HK_NEXT_MULTIPLE_OF(16, hkUint16(shapeSize) );
		}
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

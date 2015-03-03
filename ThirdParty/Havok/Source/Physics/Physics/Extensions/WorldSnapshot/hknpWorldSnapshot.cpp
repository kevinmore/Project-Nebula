/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/WorldSnapshot/hknpWorldSnapshot.h>

#include <Common/Base/Container/BlockStream/Allocator/hkBlockStreamAllocator.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>
#include <Physics/Physics/Dynamics/Constraint/hknpConstraint.h>


struct hknpWorldSnapshotImpl
{
	/// Restores the state of the specified body based on the state of a body serialized under the specified index.
	static void synchronizeBody( const hknpBody& sourceBody, hknpBody& destBody )
	{
		destBody.setTransformComAndLookAhead( sourceBody.getTransform() );
		destBody.m_aabb = sourceBody.m_aabb;
		destBody.m_collisionFilterInfo = sourceBody.m_collisionFilterInfo;
		destBody.m_qualityId = sourceBody.m_qualityId;
		destBody.m_radiusOfComCenteredBoundingSphere = sourceBody.m_radiusOfComCenteredBoundingSphere;
		destBody.m_maxTimDistance = sourceBody.m_maxTimDistance;
		destBody.m_timAngle = sourceBody.m_timAngle;
		destBody.m_maxContactDistance = sourceBody.m_maxContactDistance;
		destBody.m_motionToBodyRotation = sourceBody.m_motionToBodyRotation;
		destBody.m_userData = sourceBody.m_userData;
		destBody.setShape(sourceBody.m_shape);
		destBody.m_materialId = sourceBody.m_materialId;
	}

	/// Restores the state of the specified motion based on the state of a motion serialized under the specified index.
	static void synchronizeMotion( const hknpMotion& sourceMotion, hknpMotion& destMotion )
	{
		hknpSolverId originalSolverId = destMotion.m_solverId;
		hknpBodyId originalFirstAttachedBody = destMotion.m_firstAttachedBodyId;

		destMotion = sourceMotion;

		destMotion.m_firstAttachedBodyId = originalFirstAttachedBody;
		destMotion.m_solverId = originalSolverId;
	}
};


hknpWorldSnapshot::hknpWorldSnapshot( const hknpWorld& world )
{
	// save the world info
	world.getCinfo( m_worldCinfo );
	if ( m_worldCinfo.m_persistentStreamAllocator )
	{
		m_worldCinfo.m_persistentStreamAllocator->addReference();
	}
	if ( m_worldCinfo.m_materialLibrary )
	{
		m_worldCinfo.m_materialLibrary->addReference();
	}
	if ( m_worldCinfo.m_motionPropertiesLibrary )
	{
		m_worldCinfo.m_motionPropertiesLibrary->addReference();
	}
	if ( m_worldCinfo.m_qualityLibrary )
	{
		m_worldCinfo.m_qualityLibrary->addReference();
	}

	// Create a map from motion ids in the world to indices in the snapshot's motion array
	hkArray<hknpMotionId> newMotionIds;
	newMotionIds.setSize( world.m_motionManager.getCapacity() );
	for( int i = 0; i < newMotionIds.getSize(); i++ )
	{
		newMotionIds[i] = hknpMotionId::STATIC;
	}

	// save the bodies and motions
	{
		for( hknpBodyIterator it = world.getBodyIterator(); it.isValid(); it.next() )
		{
			const hknpBody& body = it.getBody();
			if( body.isAddedToWorld() )
			{
				m_bodies.pushBack( body );
				m_bodyNames.pushBack( hkStringPtr( world.m_bodyManager.getBodyName( body.m_id ) ) );

				// Save the body's motion. Static motion is created at world initialization and doesn't need to be saved
				hkUint32 motionIdInWorld = body.m_motionId.value();
				if (motionIdInWorld != hknpMotionId::STATIC)
				{
					if (newMotionIds[motionIdInWorld] == hknpMotionId::STATIC)
					{
						// When loaded, the IDs will be offset by one by the static motion
						newMotionIds[motionIdInWorld] = hknpMotionId(m_motions.getSize() + 1);
						m_motions.pushBack( world.m_motionManager.getMotionBuffer()[motionIdInWorld] );
					}

					// Overwrite the saved body's motion
					m_bodies.back().m_motionId = newMotionIds[motionIdInWorld];
				}

				hknpShape* shape = const_cast< hknpShape* >( body.m_shape );
				if ( shape )
				{
					shape->addReference();
				}
			}
		}
	}

	// save the constraints
	{
		int numConstraints = world.m_constraintAtomSolver->getNumConstraints();
		hknpConstraint** constraintsArr = world.m_constraintAtomSolver->getConstraints();

		m_constraints.expandBy( numConstraints );
		for ( int i = 0; i < numConstraints; ++i )
		{
			hknpConstraint* constraint = constraintsArr[i];

			m_constraints[i].m_bodyA = constraint->m_bodyIdA;
			m_constraints[i].m_bodyB = constraint->m_bodyIdB;
			m_constraints[i].m_constraintData = constraint->m_data;
			m_constraints[i].m_constraintData->addReference();
		}
	}
}

hknpWorldSnapshot::hknpWorldSnapshot( hkFinishLoadedObjectFlag flag )
	: hkReferencedObject( flag )
	, m_worldCinfo( flag )
	, m_bodies( flag )
	, m_bodyNames( flag )
	, m_motions( flag )
	, m_constraints( flag )
{
}

hknpWorldSnapshot::~hknpWorldSnapshot()
{
	int bodiesCount = m_bodies.getSize();
	for ( int i = 0; i < bodiesCount; ++i )
	{
		hknpShape* shape = const_cast< hknpShape* >( m_bodies[i].m_shape );
		if ( shape )
		{
			shape->removeReference();
		}
	}

	m_bodies.clear();
	m_bodyNames.clear();
	m_motions.clear();

	if ( m_worldCinfo.m_persistentStreamAllocator )
	{
		m_worldCinfo.m_persistentStreamAllocator->removeReference();
	}
	if ( m_worldCinfo.m_materialLibrary )
	{
		m_worldCinfo.m_materialLibrary->removeReference();
	}
	if ( m_worldCinfo.m_motionPropertiesLibrary )
	{
		m_worldCinfo.m_motionPropertiesLibrary->removeReference();
	}
	if ( m_worldCinfo.m_qualityLibrary )
	{
		m_worldCinfo.m_qualityLibrary->removeReference();
	}

	for (int i = 0; i < m_constraints.getSize(); ++i)
	{
		m_constraints[i].m_constraintData->removeReference();
	}

	m_constraints.clear();
}

void hknpWorldSnapshot::save( const char* filename ) const
{
	hkSerializeUtil::save( this, filename );
}

hknpWorld* hknpWorldSnapshot::createWorld( const hknpWorldCinfo& worldCinfo ) const
{
	// Create the world
	hknpWorld* world = new hknpWorld( worldCinfo );

	// prepare info stubs
	hknpMotionCinfo		nullMotionInfo;
	hknpBodyCinfo		nullBodyInfo;

	{
		hkVector4 defaultShapeHalfExtents;
		defaultShapeHalfExtents.set( 1.0f, 1.0f, 1.0f );
		nullBodyInfo.m_shape = hknpConvexShape::createFromHalfExtents( defaultShapeHalfExtents );
	}

	// restore motions
	{
		const int numMotions = m_motions.getSize();
		for ( int i = 0; i < numMotions; ++i )
		{
			hknpMotionId newMotionId = world->createMotion( nullMotionInfo );
			HK_ASSERT(0x5bb8a057 , newMotionId.value() == hkUint32(i + 1)   );

			hknpMotion& createdMotion = world->accessMotion( newMotionId );
			hknpWorldSnapshotImpl::synchronizeMotion( m_motions[i], createdMotion );

			// If the world cinfo was overridden and set to single threaded, override the motion's cell index.
			
			if( ( &worldCinfo != &m_worldCinfo ) &&
				( worldCinfo.m_simulationType == hknpWorldCinfo::SIMULATION_TYPE_SINGLE_THREADED ) )
			{
				createdMotion.m_cellIndex = 0;
			}
		}
	}

	// restore the bodies
	{
		const int numBodies = m_bodies.getSize();
		for ( int i = 0; i < numBodies; ++i )
		{
			const hknpBody& serializedBody = m_bodies[i];

			nullBodyInfo.m_motionId = serializedBody.m_motionId;

			HK_ON_DEBUG( hkResult result = ) world->m_bodyManager.allocateBody( serializedBody.m_id );
			HK_ASSERT2( 0x0a63b217, result == HK_SUCCESS, "Could not restore original body ID." );

			nullBodyInfo.m_reservedBodyId = serializedBody.m_id;
			hknpWorld::AdditionFlags flags = 0;
			if( !serializedBody.m_shape )
			{
				HK_WARN( 0x0a63b218, "Body " << serializedBody.m_id.value() << " has a NULL (unserialized?) shape. Skipping." );
				flags.orWith( hknpWorld::DO_NOT_ADD_BODY );
			}
			hknpBodyId newBodyId = world->createBody( nullBodyInfo, flags );

			HK_ASSERT( 0x5bc3ceb9, newBodyId == serializedBody.m_id );

			hknpBody& newBody = world->m_bodyManager.accessBody( newBodyId );

			if( m_bodyNames[i].cString() != HK_NULL )
			{
				world->m_bodyManager.setBodyName( newBodyId, m_bodyNames[i].cString() );
			}

			// Synchronize the state of the created body
			hknpWorldSnapshotImpl::synchronizeBody( serializedBody, newBody );

			// Keep a reference to the shape in the world.
			// Otherwise the shape could be destroyed when the snapshot will be.
			world->m_userReferencedObjects.pushBack( newBody.m_shape );
		}
	}

	// restore the constraints
	{
		int constraintsCount = m_constraints.getSize();
		for ( int i = 0; i < constraintsCount; ++i )
		{
			const hknpConstraintCinfo& constraintInfo = m_constraints[i];
			hknpConstraint* constraint = new hknpConstraint(constraintInfo);
			world->addConstraint( constraint );
			constraint->removeReference();
		}
	}

	// clean up
	nullBodyInfo.m_shape->removeReference();

	world->checkConsistency();
	return world;
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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSystem.h>

#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>
#include <Physics/Physics/Collide/Filter/Group/hknpGroupCollisionFilter.h>


namespace
{
	// This method is simply provided to be shared between the two disableCollisionsBetweenConstraintBodies functions below.
	// It is meant to be called in succession for all the constrained bodies.
	static void updateConstraintBodiesCollisionFilters( hkUint32& childCollisionFilterInfo, hkUint32& parentCollisionFilterInfoB, int groupFilterSystemGroup, int& subSystemId )
	{
		int subIdA = hknpGroupCollisionFilter::getSubSystemIdFromFilterInfo( childCollisionFilterInfo );
		int subIdB = hknpGroupCollisionFilter::getSubSystemIdFromFilterInfo( parentCollisionFilterInfoB );

		int ignoreA = hknpGroupCollisionFilter::getSubSystemDontCollideWithFromFilterInfo( childCollisionFilterInfo );
		int ignoreB = hknpGroupCollisionFilter::getSubSystemDontCollideWithFromFilterInfo( parentCollisionFilterInfoB );

		int layerA = hknpGroupCollisionFilter::getLayerFromFilterInfo( childCollisionFilterInfo );
		int layerB = hknpGroupCollisionFilter::getLayerFromFilterInfo( parentCollisionFilterInfoB );

		// assign subsystem ids
		if ( !subIdA ){ subIdA = subSystemId++; }
		if ( !subIdB ){ subIdB = subSystemId++; }

		if ( !ignoreA )
		{
			ignoreA = subIdB;
		}
		else
		{
			HK_ASSERT2( 0xf01a2e3f, !ignoreB, "The constraints you passed in do not form a hierarchy or are not sorted by a hierarchy" );
			ignoreB = subIdA;
		}

		childCollisionFilterInfo = hknpGroupCollisionFilter::calcFilterInfo( layerA, groupFilterSystemGroup, subIdA, ignoreA );
		parentCollisionFilterInfoB = hknpGroupCollisionFilter::calcFilterInfo( layerB, groupFilterSystemGroup, subIdB, ignoreB );
	}
}	// anon namespace


//
// Physics System Data
//

hknpPhysicsSystemData::hknpPhysicsSystemData( hkFinishLoadedObjectFlag f )
:	hkReferencedObject(f)
,	m_materials(f)
,	m_motionProperties(f)
,	m_motionCinfos(f)
,	m_bodyCinfos(f)
,	m_constraintCinfos(f)
,	m_referencedObjects(f)
,	m_name(f)
{

}

hknpPhysicsSystemData::~hknpPhysicsSystemData()
{

}

void hknpPhysicsSystemData::addBodies( const hknpWorld* world, const hknpBodyId* bodyIds, int numBodyIds )
{
	// Maps of local IDs to world IDs
	hkArray<hknpMaterialId>::Temp materialIdMap;
	hkArray<hknpMotionPropertiesId>::Temp motionPropertiesIdMap;
	hkArray<hknpMotionId>::Temp motionIdMap;
	hkArray<hknpBodyId>::Temp bodyIdMap;

	for( int i = 0; i < numBodyIds; i++ )
	{
		const hknpBody& body = world->getBody( bodyIds[i] );

		// Body
		hknpBodyCinfo& bodyCinfo = m_bodyCinfos.expandOne();
		world->getBodyCinfo( body.m_id, bodyCinfo );
		m_referencedObjects.pushBack(bodyCinfo.m_shape);
		bodyIdMap.pushBack( body.m_id );

		// Material (may be shared)
		{
			int localId = materialIdMap.indexOf( body.m_materialId );
			if( localId == -1 )
			{
				localId = m_materials.getSize();
				m_materials.pushBack( world->getMaterialLibrary()->getEntry( body.m_materialId ) );
				materialIdMap.pushBack( body.m_materialId );
			}
			bodyCinfo.m_materialId = hknpMaterialId( localId );
		}

		if( body.isDynamic() )
		{
			// Motion (may be shared)
			int localId = motionIdMap.indexOf( body.m_motionId );
			if( localId == -1 )
			{
				localId = m_motionCinfos.getSize();
				const hknpMotion& motion = world->getMotion( body.m_motionId );
				world->getMotionCinfo( body.m_motionId, m_motionCinfos.expandOne() );
				motionIdMap.pushBack( body.m_motionId );

				// Motion properties (may be shared)
				int localMpId = motionPropertiesIdMap.indexOf( motion.m_motionPropertiesId );
				if( localMpId == -1 )
				{
					localMpId = m_motionProperties.getSize();
					m_motionProperties.expandOne() = world->getMotionPropertiesLibrary()->getEntry( motion.m_motionPropertiesId );
					motionPropertiesIdMap.pushBack( motion.m_motionPropertiesId );
				}
				m_motionCinfos.back().m_motionPropertiesId = hknpMotionPropertiesId( localMpId );
			}
			bodyCinfo.m_motionId = hknpMotionId( localId );
		}
		else
		{
			bodyCinfo.m_motionId = hknpMotionId::invalid();
		}
	}

	// Find all constraints using the selected bodies
	const hkUint32 numConstraints = world->m_constraintAtomSolver->getNumConstraints();
	for( hkUint32 i = 0; i < numConstraints; i++ )
	{
		const hknpConstraint* constraint = world->m_constraintAtomSolver->getConstraints()[i];
		const int bodyAIndex = bodyIdMap.indexOf( constraint->m_bodyIdA );
		const int bodyBIndex = bodyIdMap.indexOf( constraint->m_bodyIdB );
		if( bodyAIndex >= 0 && bodyBIndex >= 0 )
		{
			hknpConstraintCinfo& cinfo = m_constraintCinfos.expandOne();
			cinfo.m_constraintData = constraint->m_data;
			cinfo.m_bodyA = hknpBodyId(bodyAIndex);
			cinfo.m_bodyB = hknpBodyId(bodyBIndex);
			cinfo.m_constraintData->addReference();
		}
	}

	checkConsistency();
}

void hknpPhysicsSystemData::addWorld( const hknpWorld* world, hknpBodyFlags flagsMask )
{
	hkArray<hknpBodyId>::Temp bodyIds;

	for( hknpBodyManager::BodyIterator it = world->getBodyIterator(); it.isValid(); it.next() )
	{
		if( it.getBody().isAddedToWorld() && (it.getBody().m_flags.anyIsSet(flagsMask)) )
		{
			bodyIds.pushBack( it.getBodyId() );
		}
	}

	addBodies( world, bodyIds.begin(), bodyIds.getSize() );
}

hknpBodyId hknpPhysicsSystemData::findBodyByName( const char* name ) const
{
	for( int i = 0; i < m_bodyCinfos.getSize(); ++i )
	{
		if( m_bodyCinfos[i].m_name == name )
		{
			return hknpBodyId(i);
		}
	}

	return hknpBodyId::invalid();
}

void hknpPhysicsSystemData::checkConsistency() const
{
#ifdef HK_DEBUG
	hkBitField materialBits( m_materials.getSize(), hkBitFieldValue::ZERO );
	hkBitField motionPropertiesBits( m_motionProperties.getSize(), hkBitFieldValue::ZERO );
	hkBitField motionBits( m_motionCinfos.getSize(), hkBitFieldValue::ZERO );

	for( int i=0; i<m_motionCinfos.getSize(); ++i )
	{
		motionPropertiesBits.set( m_motionCinfos[i].m_motionPropertiesId.value() );
	}
	for( int i=0; i<m_bodyCinfos.getSize(); ++i )
	{
		materialBits.set( m_bodyCinfos[i].m_materialId.value() );
		if( m_bodyCinfos[i].m_motionId.isValid() )
		{
			motionBits.set( m_bodyCinfos[i].m_motionId.value() );
		}

		// Checks the shapes are all referenced in the referenced objects array.
		int shapeIdx = m_referencedObjects.indexOf(m_bodyCinfos[i].m_shape);
		HK_ASSERT(0x5bf88c2a, shapeIdx != -1);
	}

	for( int i = 0; i < m_constraintCinfos.getSize(); ++i )
	{
		HK_ASSERT(0x5bf88c2b,
		m_constraintCinfos[i].m_bodyA.value() < (hkUint32)m_bodyCinfos.getSize() // check that A (child) is a bodyId from this system
		&& ( !m_constraintCinfos[i].m_bodyB.isValid()							 // check that B (parent) is either invalid (world is parent)
			|| m_constraintCinfos[i].m_bodyB.value() < (hkUint32)m_bodyCinfos.getSize() )); // or a bodyId from this system.
	}

	HK_ASSERT(0x764cca20, materialBits.getSize() == 0 || materialBits.bitCount() == materialBits.getSize() );
	HK_ASSERT(0x5f97af2a, motionPropertiesBits.getSize() == 0 || motionPropertiesBits.bitCount() == motionPropertiesBits.getSize() );
	HK_ASSERT(0x232a9433, motionBits.getSize() == 0 || motionBits.bitCount() == motionBits.getSize() );
#endif	// HK_DEBUG
}

void hknpPhysicsSystemData::collapse()
{
	//
	// Motions
	//
	if ( m_motionCinfos.getSize() > 0 )
	{
		hkBitField motionBits( m_motionCinfos.getSize(), hkBitFieldValue::ZERO );
		for ( int bodyIdx = 0; bodyIdx < m_bodyCinfos.getSize(); ++ bodyIdx )
		{
			if( m_bodyCinfos[bodyIdx].m_motionId.isValid() )
			{
				motionBits.set(m_bodyCinfos[bodyIdx].m_motionId.value());
			}
		}
		int numMotionsToDelete = motionBits.getSize() - motionBits.bitCount();
		if ( numMotionsToDelete > 0 )
		{
			hkLocalArray<int> oldToNewMap(m_motionCinfos.getSize());
			hkArray<hknpMotionCinfo> newMotionCinfos;
			for (int motionIdx = 0 ; motionIdx < motionBits.getSize(); ++motionIdx)
			{
				if ( motionBits.get(motionIdx) == 0 )
				{
					oldToNewMap.pushBack(-1);
				}
				else
				{
					newMotionCinfos.pushBack(m_motionCinfos[motionIdx]);
					oldToNewMap.pushBack(newMotionCinfos.getSize()-1);
				}
			}
			m_motionCinfos = newMotionCinfos;
			// Now fixup the remaining motionIds that changed.
			for ( int bodyIdx = 0 ; bodyIdx < m_bodyCinfos.getSize(); ++bodyIdx )
			{
				hknpBodyCinfo & bodyCinfo = m_bodyCinfos[bodyIdx];
				if ( bodyCinfo.m_motionId.isValid())
				{
					HK_ASSERT(0x320a49f, oldToNewMap[ bodyCinfo.m_motionId.value() ] != -1);
					bodyCinfo.m_motionId = hknpMotionId(oldToNewMap[ bodyCinfo.m_motionId.value() ]);
				}
			}
		}
	}

	//
	// Materials
	//
	if ( m_materials.getSize() > 0 )
	{
		hkBitField materialBits( m_materials.getSize(), hkBitFieldValue::ZERO );
		for ( int bodyIdx = 0; bodyIdx < m_bodyCinfos.getSize(); ++bodyIdx )
		{
			if( m_bodyCinfos[bodyIdx].m_materialId.isValid() )
			{
				materialBits.set(m_bodyCinfos[bodyIdx].m_materialId.value());
			}
		}
		int numMaterialsToDelete = materialBits.getSize() - materialBits.bitCount();
		if ( numMaterialsToDelete > 0 )
		{
			hkLocalArray<int> oldToNewMap(m_materials.getSize());
			hkArray<hknpMaterial> newMaterials;
			for (int materialIdx = 0 ; materialIdx < materialBits.getSize(); ++materialIdx)
			{
				if ( materialBits.get(materialIdx) == 0 )
				{
					oldToNewMap.pushBack(-1);
				}
				else
				{
					newMaterials.pushBack(m_materials[materialIdx]);
					oldToNewMap.pushBack(newMaterials.getSize()-1);
				}
			}
			m_materials= newMaterials;
			// Now fixup the remaining materialIds that changed.
			for ( int bodyIdx = 0 ; bodyIdx < m_bodyCinfos.getSize(); ++bodyIdx )
			{
				hknpBodyCinfo & bodyCinfo = m_bodyCinfos[bodyIdx];
				if ( bodyCinfo.m_materialId.isValid())
				{
					HK_ASSERT(0x320a49f, oldToNewMap[ bodyCinfo.m_materialId.value() ] != -1);
					bodyCinfo.m_materialId = hknpMaterialId(oldToNewMap[ bodyCinfo.m_materialId.value() ]);
				}
			}
		}
	}

	//
	// Motion Properties
	//
	if ( m_motionProperties.getSize() > 0 )
	{
		hkBitField motionPropertiesBits( m_motionProperties.getSize(), hkBitFieldValue::ZERO );
		for ( int motionIdx = 0; motionIdx < m_motionCinfos.getSize(); ++motionIdx )
		{
			if( m_motionCinfos[motionIdx].m_motionPropertiesId.isValid() )
			{
				motionPropertiesBits.set(m_motionCinfos[motionIdx].m_motionPropertiesId.value());
			}
		}
		int numMotionPropertiesToDelete =  motionPropertiesBits.getSize() - motionPropertiesBits.bitCount();

		if ( numMotionPropertiesToDelete > 0 )
		{
			hkLocalArray<int> oldToNewMap(m_motionProperties.getSize());
			hkArray<hknpMotionProperties> newMotionProperties;
			for (int motionPropertiesIdx = 0 ; motionPropertiesIdx < motionPropertiesBits.getSize(); ++motionPropertiesIdx)
			{
				if ( motionPropertiesBits.get(motionPropertiesIdx) == 0 )
				{
					oldToNewMap.pushBack(-1);
				}
				else
				{
					newMotionProperties.pushBack(m_motionProperties[motionPropertiesIdx]);
					oldToNewMap.pushBack(newMotionProperties.getSize()-1);
				}
			}
			m_motionProperties = newMotionProperties;
			// Now fixup the remaining motionPropertiesIds that changed.
			for ( int motionIdx = 0 ; motionIdx < m_motionCinfos.getSize(); ++motionIdx )
			{
				hknpMotionCinfo& motionCinfo = m_motionCinfos[motionIdx];
				if ( motionCinfo.m_motionPropertiesId.isValid())
				{
					HK_ASSERT(0x320a49f, oldToNewMap[ motionCinfo.m_motionPropertiesId.value() ] != -1);
					motionCinfo.m_motionPropertiesId = hknpMotionPropertiesId(oldToNewMap[ motionCinfo.m_motionPropertiesId.value() ]);
				}
			}
		}
	}
	//
	// Referenced objects : add reference to unreferenced shapes.
	//
	if ( m_referencedObjects.getSize() > 0 )
	{
		for ( int bodyIdx = 0; bodyIdx < m_bodyCinfos.getSize(); ++bodyIdx )
		{
			int shapeIdx = m_referencedObjects.indexOf(m_bodyCinfos[bodyIdx].m_shape);
			if ( shapeIdx == -1 )
			{
				m_referencedObjects.pushBack(m_bodyCinfos[bodyIdx].m_shape);
			}
		}
	}
}

void hknpPhysicsSystemData::disableCollisionsBetweenConstraintBodies( int groupFilterSystemGroup )
{
	HK_ASSERT2( 0xf021d53a, m_constraintCinfos.getSize() < 32, "The group filter allows a maximum of 32 sub ids" );

	int subSystemId = 0;
	for ( int i = 0; i < m_constraintCinfos.getSize(); ++i )
	{
		hknpBodyCinfo& child = m_bodyCinfos[m_constraintCinfos[i].m_bodyA.value()];
		hknpBodyCinfo& parent = m_bodyCinfos[m_constraintCinfos[i].m_bodyB.value()];

		updateConstraintBodiesCollisionFilters( child.m_collisionFilterInfo, parent.m_collisionFilterInfo, groupFilterSystemGroup, subSystemId );
	}
}

void hknpPhysicsSystemData::removeMotionProperty(hknpMotionPropertiesId propertyId)
{
	if ( !propertyId.isValid() )
	{
		return;
	}

	// We'll replace the property at propertyIdx with the last one
	const int propertyIdx = propertyId.value();
	const hknpMotionPropertiesId oldPropertyId(m_motionProperties.getSize() - 1);

	for (int k = m_motionCinfos.getSize() - 1; k >= 0; k--)
	{
		hknpMotionCinfo& cInfo = m_motionCinfos[k];
		if ( cInfo.m_motionPropertiesId == oldPropertyId )
		{
			cInfo.m_motionPropertiesId = propertyId;
		}
	}
	m_motionProperties.removeAt(propertyIdx);
}

void hknpPhysicsSystemData::removeMaterial(hknpMaterialId materialId)
{
	if ( !materialId.isValid() )
	{
		return;
	}

	// We'll replace the material at materialIdx with the last one
	const int materialIdx = materialId.value();
	const hknpMaterialId oldMaterialId(m_materials.getSize() - 1);

	for (int k = m_bodyCinfos.getSize() - 1; k >= 0; k--)
	{
		hknpBodyCinfo& cInfo = m_bodyCinfos[k];
		if ( cInfo.m_materialId == oldMaterialId )
		{
			cInfo.m_materialId = materialId;
		}
	}
	m_materials.removeAt(materialIdx);
}

void hknpPhysicsSystemData::removeMotion(hknpMotionId motionId)
{
	if ( !motionId.isValid() )
	{
		return;
	}

	// We'll replace the motion at motionIdx with the last one
	const int motionIdx = motionId.value();
	const hknpMotionCinfo deletedMotion = m_motionCinfos[motionIdx];
	const hknpMotionId oldMotionId(m_motionCinfos.getSize() - 1);

	for (int k = m_bodyCinfos.getSize() - 1; k >= 0; k--)
	{
		hknpBodyCinfo& cInfo = m_bodyCinfos[k];
		if ( cInfo.m_motionId == oldMotionId )
		{
			cInfo.m_motionId = motionId;
		}
	}
	m_motionCinfos.removeAt(motionId.value());

	// See if the motion property is still in use or we can delete it!
	{
		int k = m_motionCinfos.getSize() - 1;
		for (; k >= 0; k--)
		{
			if ( m_motionCinfos[k].m_motionPropertiesId == deletedMotion.m_motionPropertiesId )
			{
				break;
			}
		}

		if ( k < 0 )
		{
			// Motion property no longer used, delete it!
			removeMotionProperty(deletedMotion.m_motionPropertiesId);
		}
	}
}

bool hknpPhysicsSystemData::isUnique(hknpMotionId motionId) const
{
	// Count the number of rigid bodies using this motion Id
	int count = 0;
	for (int bi = m_bodyCinfos.getSize() - 1; bi >= 0; bi--)
	{
		if ( m_bodyCinfos[bi].m_motionId == motionId )
		{
			count++;
		}
	}
	return count < 2;
}

hknpMotionId hknpPhysicsSystemData::addMotion(const hknpMotionCinfo& newMotion)
{
	hknpMotionId retId(m_motionCinfos.getSize());
	m_motionCinfos.pushBack(newMotion);
	return retId;
}

void hknpPhysicsSystemData::removeBody(hknpBodyId bodyId)
{
	// Remove body cInfo
	const int bodyIdx = bodyId.value();
	const hknpBodyCinfo deletedBodyInfo = m_bodyCinfos[bodyIdx];
	m_bodyCinfos.removeAt(bodyIdx);

	// Check whether the motion is still used or we can delete it as well
	{
		int bi = m_bodyCinfos.getSize() - 1;
		for (; bi >= 0; bi--)
		{
			if ( m_bodyCinfos[bi].m_motionId == deletedBodyInfo.m_motionId )
			{
				break;
			}
		}

		// The motion is no longer used, we must delete it!
		if ( bi < 0 )
		{
			removeMotion(deletedBodyInfo.m_motionId);
		}
	}

	// Check whether the material is still used or we can delete it as well
	{
		int bi = m_bodyCinfos.getSize() - 1;
		for (; bi >= 0; bi--)
		{
			if ( m_bodyCinfos[bi].m_materialId == deletedBodyInfo.m_materialId )
			{
				break;
			}
		}

		// The material is no longer used, we must delete it!
		if ( bi < 0 )
		{
			removeMaterial(deletedBodyInfo.m_materialId);
		}
	}
}


//
// Physics System
//

hknpPhysicsSystem::hknpPhysicsSystem(
	const hknpPhysicsSystemData* data, hknpWorld* world, const hkTransform& transform, hknpWorld::AdditionFlags additionFlags, Flags flags )
	:	m_data( data )
	,	m_world( world )
{
	// Maps of local to world IDs
	hkArray<hknpMaterialId>::Temp materialIdMap;
	hkArray<hknpMotionPropertiesId>::Temp motionPropertiesIdMap;
	hkArray<hknpMotionId>::Temp motionIdMap;
	hkArray<hknpBodyId>& bodyIdMap = m_bodyIds;

	// create materials
	{
		const int num = m_data->m_materials.getSize();
		materialIdMap.setSize(num);
		for( int i=0; i<num; ++i )
		{
			materialIdMap[i] = m_world->accessMaterialLibrary()->addEntry( m_data->m_materials[i] );
		}
	}

	// create motions and motion properties
	{
		const int num = m_data->m_motionCinfos.getSize();
		motionIdMap.setSize(num);
		for( int i=0; i<num; ++i )
		{
			hknpMotionCinfo motionCinfo = m_data->m_motionCinfos[i];
			{
				if ( motionCinfo.m_motionPropertiesId.isValid() )
				{
					// The stored m_motionPropertiesId is an index into our created motion properties
					motionCinfo.m_motionPropertiesId = m_world->accessMotionPropertiesLibrary()->addEntry( m_data->m_motionProperties[ motionCinfo.m_motionPropertiesId.value() ] );
				}
				else
				{
					motionCinfo.m_motionPropertiesId = hknpMotionPropertiesId::KEYFRAMED; 
				}
			}

			motionIdMap[i] = m_world->createMotion( motionCinfo );
		}
	}

	// create bodies
	{
		const int num = m_data->m_bodyCinfos.getSize();
		bodyIdMap.setSize(num);
		for( int i=0; i<num; ++i )
		{
			hknpBodyCinfo bodyCinfo = m_data->m_bodyCinfos[i];
			{
				bodyCinfo.m_flags.clear( hknpBody::INTERNAL_FLAGS_MASK );
				bodyCinfo.m_materialId = materialIdMap[ bodyCinfo.m_materialId.value() ];
			}

			if( bodyCinfo.m_motionId.isValid() )
			{
				bodyCinfo.m_motionId = motionIdMap[bodyCinfo.m_motionId.value()];
			}
			else
			{
				bodyCinfo.m_motionId = hknpMotionId::STATIC;
			}

			hknpBodyId bodyId = hknpBodyId::invalid();

			// Check if the body is supposed to be added to the world
			if( !bodyCinfo.m_flags.anyIsSet(hknpBody::IS_NON_RUNTIME) && bodyCinfo.m_shape )
			{
				bodyId = m_world->createBody( bodyCinfo, additionFlags );

				// Move to desired transform
				
				{
					hkTransform bodyTransform;
					bodyTransform.set( bodyCinfo.m_orientation, bodyCinfo.m_position );
					bodyTransform.setMul( transform, bodyTransform );
					world->setBodyTransform( bodyId, bodyTransform );
				}
			}
			else if( bodyCinfo.m_motionId.isValid() && (bodyCinfo.m_motionId != hknpMotionId::STATIC) )
			{
				// Failed to create the body, free the motion
				m_world->destroyMotions(&bodyCinfo.m_motionId, 1);
			}

			bodyIdMap[i] = bodyId;
		}
	}

	// Create constraints
	{
		m_constraints.reserve( m_data->m_constraintCinfos.getSize() );
		for( int i=0; i<m_data->m_constraintCinfos.getSize(); ++i )
		{
			const hknpConstraintCinfo& cinfo = m_data->m_constraintCinfos[i];

			// Update the body ID with world body ID instead of internal.
			// An invalid ID means that the parent body is the world's default fixed body.
			const hknpBodyId bodyIdA = bodyIdMap[cinfo.m_bodyA.value()];
			const hknpBodyId bodyIdB = cinfo.m_bodyB.isValid() ? bodyIdMap[cinfo.m_bodyB.value()] : hknpBodyId::WORLD;

			if( bodyIdA.isValid() && bodyIdB.isValid() )
			{
				hkpConstraintData* constraintData = cinfo.m_constraintData;

				hkRefPtr<hkpConstraintData> clonedConstraintData;
				if( flags & CLONE_POWERABLE_CONSTRAINT_DATAS )
				{
					clonedConstraintData.setAndDontIncrementRefCount( hkpConstraintDataUtils::cloneIfCanHaveMotors( constraintData ) );
					if( clonedConstraintData.val() )
					{
						constraintData = clonedConstraintData.val();
					}
				}

				const int constraintFlags = ( flags & FORCE_EXPORTABLE_CONSTRAINTS ) ?
					hknpConstraint::IS_EXPORTABLE : hknpConstraint::NO_FLAGS;

				// Create the constraint
				hknpConstraint* constraint = new hknpConstraint();
				constraint->init( bodyIdA, bodyIdB, constraintData, (hknpConstraint::FlagBits)constraintFlags );
				m_constraints.pushBack( constraint );

				// If we are not adding bodies to the world, don't add constraints either
				if( additionFlags.get(hknpWorld::DO_NOT_ADD_BODY) == 0 )
				{
					world->addConstraint( constraint );
				}
			}
			else
			{
				m_constraints.pushBack( HK_NULL );
			}
		}
	}

	// Physics system always listen to destroyed events.
	HK_SUBSCRIBE_TO_SIGNAL( m_world->m_signals.m_bodyDestroyed, this, hknpPhysicsSystem );
}

hknpPhysicsSystem::~hknpPhysicsSystem()
{
	// Unsubscribe from destroyed signal
	m_world->m_signals.m_bodyDestroyed.unsubscribeAll( this );

	// Remove and destroy all valid constraints
	{
		for( int i = 0; i < m_constraints.getSize(); ++i )
		{
			if( m_constraints[i] )
			{
				if( m_world->isConstraintAdded( m_constraints[i] ) )
				{
					m_world->removeConstraint( m_constraints[i] );
				}

				// Destroy it (unless referenced elsewhere)
				m_constraints[i]->removeReference();
			}
		}
	}

	// Remove and destroy all valid bodies
	{
		hkLocalArray<hknpBodyId> bodiesToDestroy( m_bodyIds.getSize() );
		for( int i = 0; i < m_bodyIds.getSize(); ++i )
		{
			if( m_bodyIds[i].isValid() )
			{
				bodiesToDestroy.pushBackUnchecked( m_bodyIds[i] );
			}
		}

		// This removes any added bodies too
		m_world->destroyBodies( bodiesToDestroy.begin(), bodiesToDestroy.getSize() );
	}
}

void hknpPhysicsSystem::addToWorld( hknpWorld::AdditionFlags bodyAdditionFlags,
	hknpActivationMode::Enum constraintActivationMode )
{
	// Add any valid bodies that are not already added to the world
	{
		hkLocalArray<hknpBodyId> bodiesToAdd( m_bodyIds.getSize() );
		for( int i = 0; i < m_bodyIds.getSize(); ++i )
		{
			if( m_bodyIds[i].isValid() && !m_world->isBodyAdded(m_bodyIds[i]) )
			{
				bodiesToAdd.pushBackUnchecked( m_bodyIds[i] );
			}
		}

		m_world->addBodies( bodiesToAdd.begin(), bodiesToAdd.getSize(), bodyAdditionFlags );
	}

	// Add any valid constraints between added bodies that are not already added to the world
	for( int i = 0; i < m_constraints.getSize(); ++i )
	{
		const hknpConstraint* constraint = m_constraints[i];
		if( constraint && !m_world->isConstraintAdded( constraint ) )
		{
			if( m_world->isBodyAdded(constraint->m_bodyIdA) &&
				m_world->isBodyAdded(constraint->m_bodyIdB) )
			{
				m_world->addConstraint( m_constraints[i], constraintActivationMode );
			}
		}
	}
}

void hknpPhysicsSystem::removeFromWorld()
{
	// Remove all added constraints
	{
		for( int i = 0; i < m_constraints.getSize(); ++i )
		{
			if( m_constraints[i] && m_world->isConstraintAdded(m_constraints[i]) )
			{
				m_world->removeConstraint( m_constraints[i] );
			}
		}
	}

	// Remove all added bodies
	{
		hkLocalArray<hknpBodyId> bodiesToRemove( m_bodyIds.getSize() );
		for( int i = 0; i < m_bodyIds.getSize(); ++i )
		{
			if( m_bodyIds[i].isValid() && m_world->isBodyAdded(m_bodyIds[i]) )
			{
				bodiesToRemove.pushBackUnchecked( m_bodyIds[i] );
			}
		}

		m_world->removeBodies( bodiesToRemove.begin(), bodiesToRemove.getSize() );
	}
}

void hknpPhysicsSystem::disableCollisionsBetweenConstraintBodies( int groupFilterSystemGroup )
{
	HK_ASSERT2( 0xf021d53a, m_bodyIds.getSize() < 31, "The group filter allows a maximum of 32 sub ids" );

	int subSystemId = 0;
	for ( int i = 0; i <  m_constraints.getSize(); ++i )
	{
		const hknpBodyId child = m_constraints[i]->m_bodyIdA;
		const hknpBodyId parent = m_constraints[i]->m_bodyIdB;
		hkUint32 childCollisionFilterInfo = m_world->getBody( child ).m_collisionFilterInfo;
		hkUint32 parentCollisionFilterInfo = m_world->getBody( parent ).m_collisionFilterInfo;

		updateConstraintBodiesCollisionFilters( childCollisionFilterInfo, parentCollisionFilterInfo, groupFilterSystemGroup, subSystemId );

		m_world->setBodyCollisionFilterInfo( child, childCollisionFilterInfo );
		m_world->setBodyCollisionFilterInfo( parent, parentCollisionFilterInfo );
	}
}

void hknpPhysicsSystem::onBodyDestroyedSignal( hknpWorld* world, hknpBodyId bodyId )
{
	// Body is destroyed and marked as invalid.
	// Don't remove it since we want to maintain a one-to-one mapping with the bodyCinfos in the system data.
	HK_ASSERT(0x34004a1, world == m_world );
	int i = m_bodyIds.indexOf( bodyId );
	if( i >= 0 )
	{
		m_bodyIds[i] = hknpBodyId::invalid();
		
	}
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

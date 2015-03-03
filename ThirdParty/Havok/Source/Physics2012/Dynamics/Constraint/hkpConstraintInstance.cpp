/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintOwner.h>

// for cloning:
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintCallbackUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintDataCloningUtil.h>
#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>

#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>
#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics/Constraint/Data/Fixed/hkpFixedConstraintData.h>
#include <Physics/Constraint/Data/DeformableFixed/hkpDeformableFixedConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Malleable/hkpMalleableConstraintData.h>
#include <Physics/Constraint/Data/PointToPath/hkpPointToPathConstraintData.h>
#include <Physics/Constraint/Data/PointToPlane/hkpPointToPlaneConstraintData.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>
#include <Physics/Constraint/Data/StiffSpring/hkpStiffSpringConstraintData.h>
#include <Physics/Constraint/Data/Wheel/hkpWheelConstraintData.h>

hkpConstraintInstance::hkpConstraintInstance(hkpEntity* entityA, hkpEntity* entityB, hkpConstraintData* data, hkpConstraintInstance::ConstraintPriority priority)
	:	m_owner(HK_NULL),
		m_data(data),
		// entities, below
		m_priority(priority),
		m_wantRuntime(true),
		m_destructionRemapInfo(ON_DESTRUCTION_REMAP),
		m_userData(0), // can call setUserData to set it.
		m_internal(HK_NULL), 
		m_uid(0xfffffff0)
{
	// setup contraintInternal members
	m_entities[0] = entityA;
	m_entities[1] = entityB;
	m_constraintModifiers = HK_NULL;

	HK_ASSERT2(0xf0fe4356, entityA, "EntityA not set.");

	//Chris: Upon addition to the world, a constraint instance has
	// its any null entity (of which only B can be null)
	// set as the world's fixed body. Thus NULL is allowed for B
	// and this happens by default in the serialization snapshot
	// and constraints created without knowledge of the world
	// have no access to a fixed body unless they make a dummy one
	// HK_ASSERT2(0xf0fe4356, entityB, "EntityB not set.");


	{
		hkReferencedObject::lockAll();
		m_entities[0]->addReference();
		if (m_entities[1] != HK_NULL) 
		{
			m_entities[1]->addReference();
		}
		m_data->addReference();
		hkReferencedObject::unlockAll();
	}
}

hkpConstraintInstance::hkpConstraintInstance(hkpConstraintInstance::ConstraintPriority priority)
:	m_owner(HK_NULL),
	m_constraintModifiers(HK_NULL),
	m_priority(priority),
	m_wantRuntime(true),
	m_destructionRemapInfo(ON_DESTRUCTION_REMAP),
	m_userData(0),
	m_internal(HK_NULL),
	m_uid(0xfffffff0)
{
}


void hkpConstraintInstance::setPriority( hkpConstraintInstance::ConstraintPriority priority ) 
{
	m_priority = priority;
	if ( m_internal )
	{
		m_internal->m_priority = priority;
	}
}

hkpSimulationIsland* hkpConstraintInstance::getSimulationIsland()
{
	if (!m_entities[0]->isFixed())
	{
		return m_entities[0]->getSimulationIsland();
	}
	return m_entities[1]->getSimulationIsland();
}


void hkpConstraintInstance::entityAddedCallback(hkpEntity* entity)
{
	HK_ASSERT2(0x11f2a0b1, 0, "Internal Error");
}

void hkpConstraintInstance::entityDeletedCallback( hkpEntity* entity )
{
	HK_ASSERT2(0x11f2a0b1, 0, "Internal Error: the constraint should be holding a reference to its bodies.");
}

void hkpConstraintInstance::entityRemovedCallback(hkpEntity* entity)
{
	
	
	if ( m_owner != HK_NULL )
	{
		HK_ASSERT2(0xad4bd4d3, entity->getWorld(), "Internal error: entity passed in hkpConstraintInstance::entityRemovedCallback is already removed from the world (Constraints must be removed first).");
		hkpWorldOperationUtil::removeConstraintImmediately(entity->getWorld(), this);
	}
}

hkpConstraintInstance* hkpConstraintInstance::clone(hkpEntity* newEntityA, hkpEntity* newEntityB, CloningMode mode) const
{
	HK_ASSERT2(0xad67888a, getType() != hkpConstraintInstance::TYPE_CHAIN, "Cloning only works for normal constraints, not for constraint chains, for example.");

	hkpConstraintInstance* instance;

	if ( mode == CLONE_SHALLOW_IF_NOT_CONSTRAINED_TO_WORLD )
	{
		
		// When body B is NULL, we perform a deep copy of the constraint.
		if ( isConstrainedToWorld() )
		{
			hkpConstraintData* deepClonedData = hkpConstraintDataCloningUtil::deepClone( getData() );
			instance = new hkpConstraintInstance( newEntityA, HK_NULL, deepClonedData, m_priority );
			deepClonedData->removeReference();
		}
		else
		{
			instance = new hkpConstraintInstance( newEntityA, newEntityB, getDataRw(), m_priority );
		}
	}
	else if ( mode == CLONE_DATAS_WITH_MOTORS )
	{
		hkpConstraintData* newData = hkpConstraintDataUtils::cloneIfCanHaveMotors( getData() );
		if (!newData)
		{
			newData = getDataRw();
			newData->addReference();
		}
		instance = new hkpConstraintInstance( newEntityA, newEntityB, newData, m_priority );
		newData->removeReference();
	}
	else //if ( mode == CLONE_FORCE_SHALLOW )
	{
		// A shallow copy of the constraint data is made, regardless of whether or not the constraint is
		// constrained to the world
		instance = new hkpConstraintInstance( newEntityA, newEntityB, getDataRw(), m_priority );
	}

	
	// The name is cloned by copying the pointer to the 'name' hkStringPtr 
	// IMPORTANT: The name is not cleaned up by the hkpConstraintInstance destructor. You are required to track it yourself.
	instance->setName( getName() );

	return instance;
}

hkpConstraintInstance::~hkpConstraintInstance()
{
	HK_ASSERT2(0x733aae9d, HK_NULL == m_owner, "hkpConstraintInstance has an owner and should not be deleted.");

	hkpConstraintCallbackUtil::fireConstraintDeleted( this );

	if (m_entities[0] != HK_NULL)
	{
		m_entities[0]->removeReference(); 
	}
	if (m_entities[1] != HK_NULL)
	{
		m_entities[1]->removeReference();
	}

	//
	//	Delete modifiers
	//
	if (m_constraintModifiers)
	{
		hkpConstraintAtom* atom = m_constraintModifiers;
		while( atom->isModifierType() )
		{
			hkpModifierConstraintAtom* modifier = reinterpret_cast<hkpModifierConstraintAtom*>(atom);
			atom = modifier->m_child;

			HK_MEMORY_TRACKER_DELETE_OBJECT(modifier)
			hkDeallocateChunk<void>(modifier, modifier->m_modifierAtomSize, HK_MEMORY_CLASS_CONSTRAINT );
		}
	}

	//
	// Only after modifiers are removed, we can destroy the data and the terminal atom.
	//
	if (m_data)
	{
		m_data->removeReference();
	}
}


void hkpConstraintInstance::addConstraintListener( hkpConstraintListener* listener )
{
	hkSmallArray<hkpConstraintListener*>& arr = m_listeners;

	HK_ASSERT2(0xf0a5ad5d, arr.indexOf( listener ) < 0, "You tried to add an entity listener twice" );
	int emptyIndex = arr.indexOf( HK_NULL );
	if ( emptyIndex >= 0)
	{
		arr[emptyIndex] = listener;
	}
	else
	{
		arr.pushBack( listener );
	}
}

void hkpConstraintInstance::removeConstraintListener( hkpConstraintListener* listener )
{
	hkSmallArray<hkpConstraintListener*>& arr = m_listeners;
	int i = arr.indexOf( listener );
	HK_ASSERT2(0x79e1d7d7, i >= 0, "You tried to remove an entity listener, which was never added" );
	arr[i] = HK_NULL;
}

void hkpConstraintInstance::transform( const hkTransform& transformation )
{			
	if ( !isConstrainedToWorld() )
	{
		return;
	}

	hkpConstraintData* data = getDataRw();

	// goto to repeat switch on data inside MALLEABLE or BREAKABLE constraint
hkpConstraintInstance_transform_data:

	HK_ASSERT2(0x0606a178, data, "Constraint data in instance not initialized.");

	switch(data->getType())
	{
	case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:
		{
			hkpBallAndSocketConstraintData* ballSocketData = static_cast<hkpBallAndSocketConstraintData*>(data);

			hkVector4& pivotA =	ballSocketData->m_atoms.m_pivots.m_translationA;
			hkVector4& pivotB =	ballSocketData->m_atoms.m_pivots.m_translationB;
			pivotB.setTransformedPos(transformation,pivotB);
			ballSocketData->setInBodySpace(pivotA,pivotB);
			break;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_FIXED:
		{
			hkpFixedConstraintData* fixedConstraintData = static_cast<hkpFixedConstraintData*>(data);

			hkTransform pivotA	= fixedConstraintData->m_atoms.m_transforms.m_transformA;
			hkTransform pivotB;	pivotB.setMul(transformation, fixedConstraintData->m_atoms.m_transforms.m_transformB);
			fixedConstraintData->setInBodySpace(pivotA, pivotB);
			break;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:
		{
			hkpDeformableFixedConstraintData* constraintData = static_cast<hkpDeformableFixedConstraintData*>(data);

			hkTransform pivotA	= constraintData->m_atoms.m_transforms.m_transformA;
			hkTransform pivotB;	pivotB.setMul(transformation, constraintData->m_atoms.m_transforms.m_transformB);
			constraintData->setInBodySpace(pivotA, pivotB);
			break;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:
		{
			hkpBreakableConstraintData* breakableData = static_cast<hkpBreakableConstraintData*>(data);

			data = breakableData->getWrappedConstraintData();
			goto hkpConstraintInstance_transform_data;
			break;
		}
	case hkpConstraintData::CONSTRAINT_TYPE_HINGE:
		{
			hkpHingeConstraintData* hingeData = static_cast<hkpHingeConstraintData*>(data);

			hkTransform aTransform = hingeData->m_atoms.m_transforms.m_transformA;
			hkTransform bTransform = hingeData->m_atoms.m_transforms.m_transformB;
			hkVector4& pivotA =	 aTransform.getTranslation();
			hkVector4& pivotB =	 bTransform.getTranslation();
			hkVector4 axisA = aTransform.getColumn<0>();
			hkVector4 axisB = bTransform.getColumn<0>();
			pivotB.setTransformedPos(transformation,pivotB);
			axisB.setRotatedDir(transformation.getRotation(), axisB);
			hingeData->setInBodySpace(pivotA,pivotB,axisA,axisB);
			break;
		}
	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
		{
			hkpLimitedHingeConstraintData* hingeData = static_cast<hkpLimitedHingeConstraintData*>(data);

			hkTransform aTransform = hingeData->m_atoms.m_transforms.m_transformA;
			hkTransform bTransform = hingeData->m_atoms.m_transforms.m_transformB;
			hkVector4 axisA = aTransform.getColumn<0>();
			hkVector4 axisB = bTransform.getColumn<0>();
			hkVector4 axisAPerp = aTransform.getColumn<1>();
			hkVector4 axisBPerp = bTransform.getColumn<1>();
			hkVector4& pivotA =	 aTransform.getColumn(3);
			hkVector4& pivotB =	 bTransform.getColumn(3);
			pivotB.setTransformedPos(transformation,pivotB);
			axisB.setRotatedDir(transformation.getRotation(), axisB);
			axisBPerp.setRotatedDir(transformation.getRotation(), axisBPerp);
			hingeData->setInBodySpace(pivotA,pivotB,axisA,axisB,axisAPerp,axisBPerp);			
			break;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:
		{
			hkpMalleableConstraintData* malleableData = static_cast<hkpMalleableConstraintData*>(data);

			data = malleableData->getWrappedConstraintData();
			goto hkpConstraintInstance_transform_data;
			break;
		}
	case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:
		{
			hkpPointToPlaneConstraintData* planeData = static_cast<hkpPointToPlaneConstraintData*>(data);

			hkVector4 pivotA = planeData->m_atoms.m_transforms.m_transformA.getTranslation();
			hkVector4 pivotB = planeData->m_atoms.m_transforms.m_transformB.getTranslation();
			int planeNormalIndex = planeData->m_atoms.m_lin.m_axisIndex;
			hkVector4 normalB = planeData->m_atoms.m_transforms.m_transformB.getColumn(planeNormalIndex);
			pivotB.setTransformedPos(transformation,pivotB);
			normalB.setRotatedDir(transformation.getRotation(),normalB);
			planeData->setInBodySpace(pivotA,pivotB,normalB);
			break;
		}
	case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:
		{
			hkpPrismaticConstraintData* prisData = static_cast<hkpPrismaticConstraintData*>(data);

			hkTransform aTransform = prisData->m_atoms.m_transforms.m_transformA;
			hkTransform bTransform = prisData->m_atoms.m_transforms.m_transformB;
			hkVector4 axisA = aTransform.getColumn<0>();
			hkVector4 axisPerpA = aTransform.getColumn<1>();
			hkVector4 pivotA = aTransform.getTranslation();
			hkVector4 axisB = bTransform.getColumn<0>();
			hkVector4 axisPerpB = bTransform.getColumn<1>();
			hkVector4 pivotB = bTransform.getTranslation();
			pivotB.setTransformedPos(transformation, pivotB);
			axisB.setRotatedDir(transformation.getRotation(), axisB);
			axisPerpB.setRotatedDir(transformation.getRotation(), axisPerpB);
			prisData->setInBodySpace(pivotA,pivotB,axisA,axisB,axisPerpA,axisPerpB);
			break;
		}
	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
		{
			hkpRagdollConstraintData* ragDollData = static_cast<hkpRagdollConstraintData*>(data);

			hkTransform aTransform = ragDollData->m_atoms.m_transforms.m_transformA;
			hkTransform bTransform = ragDollData->m_atoms.m_transforms.m_transformB;
			hkVector4* baseA = &ragDollData->m_atoms.m_transforms.m_transformA.getColumn(0);
			hkVector4& twistAxisA = baseA[ragDollData->m_atoms.AXIS_TWIST];
			hkVector4& planeAxisA = baseA[ragDollData->m_atoms.AXIS_PLANES];
			hkVector4 *baseB = &ragDollData->m_atoms.m_transforms.m_transformB.getColumn(0);
			hkVector4& twistAxisB = baseB[ragDollData->m_atoms.AXIS_TWIST];
			hkVector4& planeAxisB = baseB[ragDollData->m_atoms.AXIS_PLANES];
			hkVector4& pivotA =	 aTransform.getTranslation();
			hkVector4& pivotB =	 bTransform.getTranslation();
			pivotB.setTransformedPos(transformation,pivotB);
			planeAxisB.setRotatedDir(transformation.getRotation(),planeAxisB);
			twistAxisB.setRotatedDir(transformation.getRotation(),twistAxisB);
			ragDollData->setInBodySpace(pivotA, pivotB, planeAxisA, planeAxisB, twistAxisA, twistAxisB);
			break;
		}
	case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:
		{
			hkpStiffSpringConstraintData* springData = static_cast<hkpStiffSpringConstraintData*>(data);

			hkVector4 pivotA = springData->m_atoms.m_pivots.m_translationA;
			hkVector4 pivotB = springData->m_atoms.m_pivots.m_translationB;
			hkReal restLength = springData->m_atoms.m_spring.m_length;
			pivotB.setTransformedPos(transformation,pivotB);
			springData->setInBodySpace(pivotA,pivotB,restLength);
			break;
		}
	case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:
		{
			hkpWheelConstraintData* wheelData = static_cast<hkpWheelConstraintData*>(data);

			hkTransform aTransform = wheelData->m_atoms.m_suspensionBase.m_transformA;
			hkTransform bTransform = wheelData->m_atoms.m_suspensionBase.m_transformB;
			hkVector4 pivotA = aTransform.getTranslation();
			hkVector4 pivotB = bTransform.getTranslation();
			hkVector4 suspensionAxisB =  wheelData->m_atoms.m_suspensionBase.m_transformB.getColumn<0>();
			hkVector4 axleA = wheelData->m_atoms.m_steeringBase.m_rotationA.getColumn<0>();
			hkVector4 axleB = wheelData->m_atoms.m_steeringBase.m_rotationB.getColumn<0>();
			pivotB.setTransformedPos(transformation,pivotB);
			hkVector4 steeringAxisB =	 wheelData->m_atoms.m_steeringBase.m_rotationB.getColumn<1>();
			suspensionAxisB.setRotatedDir(transformation.getRotation(), suspensionAxisB);
			axleB.setRotatedDir(transformation.getRotation(), axleB);
			steeringAxisB.setRotatedDir(transformation.getRotation(), steeringAxisB);
			wheelData->setInBodySpace(pivotA, pivotB, axleA, axleB, suspensionAxisB, steeringAxisB);
			break;
		}
	default:
		{
			HK_WARN(0x0606a378, "Transforming this constraint is not supported");
			break;
		}
	}
}

void hkpConstraintInstance::pointNullsToFixedRigidBody()
{
	HK_ASSERT( 0x225ef5d3, m_owner == HK_NULL );
	
	for( int i = 0; i < 2; ++i )
	{
		if( m_entities[i] == HK_NULL )
		{
			hkpEntity* other = getOtherEntity(m_entities[i]);
			if( other && other->getWorld() )
			{
				m_entities[i] = other->getWorld()->getFixedRigidBody();
				m_entities[i]->addReference();
			}
		}
	}
}

hkBool hkpConstraintInstance::isConstrainedToWorld() const
{
	return ( !m_entities[1] || ( m_entities[0]->getWorld() && ( m_entities[1] == m_entities[0]->getWorld()->getFixedRigidBody() ) ) );
}

void hkpConstraintInstance::getPivotsInWorld(hkVector4& pivotAinW, hkVector4& pivotBinW) const
{
	hkVector4 pivotA, pivotB;
	pivotA = hkpConstraintDataUtils::getPivotA(m_data);
	pivotB = hkpConstraintDataUtils::getPivotB(m_data);

	const hkTransform& transformA = getEntityA()->getCollidable()->getTransform();
	const hkTransform& transformB = getEntityB()->getCollidable()->getTransform();

	pivotAinW.setTransformedPos(transformA, pivotA);
	pivotBinW.setTransformedPos(transformB, pivotB);
}

void hkpConstraintInstance::setVirtualMassInverse(const hkVector4& invMassA, const hkVector4& invMassB)
{
	HK_ASSERT2(0xdbc91737, getOwner(), "hkpConstraintInstance::setVirtualMassInverse can only be called after the constraint has been added to a world.");
	
	if (hkpModifierConstraintAtom* modifier = hkpWorldConstraintUtil::findModifier(this, hkpConstraintAtom::TYPE_MODIFIER_MASS_CHANGER))
	{
		hkpMassChangerModifierConstraintAtom* massChangermodifier = reinterpret_cast<hkpMassChangerModifierConstraintAtom*>(modifier);
		massChangermodifier->m_factorA = invMassA;
		massChangermodifier->m_factorB = invMassB;
	}
	else
	{
		hkpMassChangerModifierConstraintAtom* massChangermodifier = new hkpMassChangerModifierConstraintAtom;
		massChangermodifier->m_factorA = invMassA;
		massChangermodifier->m_factorB = invMassB;
		hkpWorldConstraintUtil::addModifier(this, *getOwner(), massChangermodifier);
	}
}

void hkpConstraintInstance::enable ( void )
{
	HK_ASSERT2(0xdbc91737, getOwner(), "hkpConstraintInstance::enable can only be called after the constraint has been added to a world.");
	hkpResponseModifier::enableConstraint(this, *getOwner());
}

void hkpConstraintInstance::disable ( void )
{
	HK_ASSERT2(0xdbc91737, getOwner(), "hkpConstraintInstance::disable can only be called after the constraint has been added to a world.");
	hkpResponseModifier::disableConstraint(this, *getOwner());
}

void hkpConstraintInstance::setEnabled ( hkBool state )
{
	if (state)
	{
		enable();
	}
	else
	{
		disable();
	}
}

hkBool hkpConstraintInstance::isEnabled ( void )
{
	return hkpWorldConstraintUtil::findModifier(this, hkpConstraintAtom::TYPE_MODIFIER_IGNORE_CONSTRAINT) == HK_NULL;
}

void hkpConstraintInstance::replaceEntity(hkpEntity* oldEntity, hkpEntity* newEntity)
{
	HK_ASSERT2(0xad63291, getOwner() == HK_NULL, "This only works when the constraint is not in a world.");

	// Exchange entity pointers
	int bodyIndex = oldEntity == m_entities[0] ? 0 : 1;
	HK_ASSERT2(0xad831134, m_entities[bodyIndex] == oldEntity, "Entity not found in the constraint.");

	newEntity->addReference();
	if (oldEntity)
	{
		oldEntity->removeReference();
	}
	m_entities[bodyIndex] = newEntity;
}

void hkpConstraintInstance::setFixedRigidBodyPointersToZero( hkpWorld* world )
{
	HK_ASSERT2( 0xf0342342, getOwner() == HK_NULL, "You can only call this function if the constraint is already removed from the world" );

	for( int i = 0; i < 2; ++i )
	{
		if( m_entities[i] == world->getFixedRigidBody() )
		{
			m_entities[i]->removeReference();
			m_entities[i] = HK_NULL;
		}
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
